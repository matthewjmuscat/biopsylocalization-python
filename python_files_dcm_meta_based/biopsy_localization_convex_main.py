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
import alphashape
import pymeshfix
import pyvista as pv
import point_containment_tools
import multiprocess
#import pathos, multiprocess
#from pathos.multiprocessing import ProcessingPool
import dill
import math
import cupy as cp
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
import pickle
import dataframe_builders
import cuspatial
import geopandas
from itertools import combinations
import biopsy_transporter
import matplotlib.pyplot as plt
from collections import defaultdict
import lattice_reconstruction_tools
import MC_prepper_funcs 
import MC_simulator_MR
import dose_lattice_helper_funcs
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import polygon_dilation_helpers_numpy
import cProfile
import pstats
import io
from line_profiler import LineProfiler
import mr_localizers
from preprocessing.interpolation.interpolation import interpolation_information_obj
from preprocessing.biopsy_processing.biopsy_processor import real_biopsy_processer
from preprocessing.biopsy_processing.biopsy_centroid_variation_validation import validate_simulated_biopsy_planned_vs_realized_centroid_variation
from preprocessing.biopsy_processing.biopsy_double_sextant import biopsy_double_sextant_processer
from preprocessing.biopsy_processing.realized_biopsy_targeting import realized_biopsy_targeting_processer
from preprocessing.biopsy_processing.sampled_biopsy_processing import sampled_biopsy_processing_processer
from preprocessing.biopsy_processing.simulated_biopsy_planner import simulated_biopsy_planner_processer
from preprocessing.biopsy_processing.simulated_biopsy_processor import simulated_biopsy_processer
from preprocessing.biopsy_processing.simulated_biopsy_preparation import simulated_biopsy_preparer
from preprocessing.biopsy_processing.simulated_biopsy_preparation import get_prepared_simulated_biopsy_length_mm
from preprocessing.transform_bank import MAX_GENERATED_TRANSFORM_SAMPLES_KEY
from preprocessing.transform_bank import OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY
from preprocessing.transform_bank import STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY
from preprocessing.transform_bank import resolve_required_generated_transform_samples
from preprocessing.uncertainty_attachment import prepare_and_attach_uncertainty_data
from preprocessing.pickled_dataset_tools import export_preprocessed_pickle_bundle
from preprocessing.pickled_dataset_tools import rebuild_loaded_preprocessed_runtime_objects
from preprocessing.pickled_dataset_tools import resolve_loaded_frozen_preprocessed_bundle_config
from preprocessing.output_runtime_dirs import create_run_output_directories
from preprocessing.render_debug_surface import render_processed_dataset_debug_processer
from preprocessing.structure_processing.non_biopsy_structure_loop import finalize_non_biopsy_structure_legacy_validation
from preprocessing.structure_processing.non_biopsy_structure_loop import prepare_non_biopsy_structure_legacy_validation
from preprocessing.structure_processing.non_biopsy_structure_loop import process_standard_non_biopsy_structure_families
from preprocessing.structure_processing.prostate_only_mr_adc import prostate_only_mr_adc_processer
from sampling import biopsy_point_sampler
from biopsy_optimizer.v1.biopsy_optimizer_module_v1 import biopsy_optimizer_module_v1
from biopsy_optimizer.v2.biopsy_optimizer_module_v2 import build_optimizer_v2_adaptive_block_search_config
from biopsy_optimizer.v2.live_integration import (
    TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY,
    TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY,
    annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit,
    annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores,
    run_target_dil_optimizer_v2_for_live_simulated_family,
)
from config import ArtifactConfig
from config import GuidanceMapConfig
from config import OptimizerRuntimeConfig
from config import PipelineConfig
from config import PreprocessingConfig
from config import RandomSeedConfig
from config import RuntimeReplayConfig
from config import RuntimeUIConfig
from guidance_maps.config import GuidanceMapPlanningConfig
from guidance_maps.planning import precompute_guidance_map_firing_depth_recommendations_for_run
from input_data import write_input_manifest_files
from output_artifacts import build_in_memory_stitch_validation
from output_artifacts import PHASE3C_OUTPUT_DIR_NAME
from output_artifacts import summarize_in_memory_stitch_validation
from output_artifacts import write_in_memory_stitch_validation_outputs
from output_artifacts import write_phase3c_output_surface
from patient_runner import DEFAULT_PATIENT_RUNNER_SHADOW_OUTPUT_DIR_NAME
from patient_runner import LegacyRuntimeKeys
from patient_runner import PatientRunnerMainValidationConfig
from patient_runner import PatientRunnerMainValidationMode
from patient_runner import run_patient_runner_main_validation
from patient_runner import summarize_patient_runner_main_validation
from startup.guidance_map_workflow import GuidanceMapRenderConfig
from startup.guidance_map_workflow import render_guidance_maps_for_run
from startup.pickle_bundle_run_loader import load_selected_pickle_bundle_run
from startup.runtime_logging import RuntimeLogger
from startup.runtime_logging import install_runtime_logger


def resolve_optimizer_v2_transform_sample_count(optimizer_v2_search_config):
    return optimizer_v2_search_config.resolve_required_transform_bank_size()


def configure_transform_precompute_settings(master_structure_info_dict,
                                            optimizer_v2_search_config,
                                            num_stochastic_targeting_transform_samples_input):
    mc_info = master_structure_info_dict["Global"].setdefault("MC info", {})
    mc_info[OPTIMIZER_V2_TRANSFORM_SAMPLE_COUNT_KEY] = resolve_optimizer_v2_transform_sample_count(optimizer_v2_search_config)
    mc_info[STOCHASTIC_TARGETING_TRANSFORM_SAMPLE_COUNT_KEY] = num_stochastic_targeting_transform_samples_input


def configure_runtime_random_seed_settings(master_structure_info_dict,
                                           transform_generation_random_seed,
                                           optimizer_v1_random_seed):
    random_info = master_structure_info_dict["Global"].setdefault("Random info", {})
    random_info["Transform generation random seed"] = transform_generation_random_seed
    random_info["Optimizer v1 random seed"] = optimizer_v1_random_seed


def build_transform_generation_rng(master_structure_info_dict):
    random_info = master_structure_info_dict["Global"].setdefault("Random info", {})
    transform_generation_random_seed = random_info.get("Transform generation random seed")
    if transform_generation_random_seed is None:
        return cp.random.RandomState()

    return cp.random.RandomState(transform_generation_random_seed)


def apply_optimizer_v1_random_seed(master_structure_info_dict):
    random_info = master_structure_info_dict["Global"].setdefault("Random info", {})
    optimizer_v1_random_seed = random_info.get("Optimizer v1 random seed")
    if optimizer_v1_random_seed is None:
        return

    cp.random.seed(optimizer_v1_random_seed)
    np.random.seed(optimizer_v1_random_seed)


def configure_transform_generation_counts(master_structure_info_dict,
                                          num_mc_containment_simulations_input,
                                          num_mc_dose_simulations_input,
                                          num_mc_mr_simulations_input):
    mc_info = master_structure_info_dict["Global"]["MC info"]

    mc_info["Num MC containment simulations"] = num_mc_containment_simulations_input
    mc_info["Num MC dose simulations"] = num_mc_dose_simulations_input
    mc_info["Num MC MR simulations"] = num_mc_mr_simulations_input

    max_num_mc_simulations = max(num_mc_dose_simulations_input,
                                 num_mc_containment_simulations_input,
                                 num_mc_mr_simulations_input)
    mc_info["Max of num MC simulations"] = max_num_mc_simulations

    _, max_generated_transform_samples = resolve_required_generated_transform_samples(
        mc_info,
        num_mc_containment_simulations_input,
        num_mc_dose_simulations_input,
        num_mc_mr_simulations_input,
    )
    mc_info[MAX_GENERATED_TRANSFORM_SAMPLES_KEY] = max_generated_transform_samples

    return max_num_mc_simulations, max_generated_transform_samples


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


    ### Non-user changeable keys 
    all_ref_key = "All ref"
    bx_ref = "Bx ref"
    by_patient_key = "By patient"
    global_key = "Global"
    global_num_cases_key = "Num cases"
    oar_ref = "OAR ref"
    dil_ref = "DIL ref"
    rectum_ref_key = "Rectum ref"
    urethra_ref_key = "Urethra ref"
    ###


    # NOTE: DONT THINK WE WANT TO INCLUDE PATIENT 198 (F1), DOES NOT HAVE ANY BIOPSIES!


    # Data removals dictionary (Specify patient and biopsy ids to remove from the dataset)
    # specify the patient IDs and the list of biopsy names to remove from the dataset
    data_removals_dict_bx = {"189 (F2)": ["Bx_Tr LM1 blood"],
                            "192 (F2)": ["Bx_trk LM blood"],
                            "200 (F1)": ["Bx_LTapex_needle"],#["Bx_LTapex_air"], # the air in this case is actually better, the needle structure is way too long in this one
                            "201 (F2)": ["Bx_LTpost_air"],
                            "203 (F1)": ["Bx_LTapex_air"],
                            }
    
    data_removals_dict_dil = {"194 (F1)": ["DIL 2"],
                            "194 (F2)": ["DIL 2"],
                            "195 (F2)": ["DIL 1 MIN", "DIL 2 MIN"],
                            "196 (F1)": ["DIL 1 MIN"],
                            "196 (F2)": ["DIL 1 MIN"],
                            "199 (F1)": ["DIL 1 MIN", "DIL 2 MIN"],
                            "199 (F2)": ["DIL 1 MIN", "DIL 2 MIN"],
                            }
    
    data_removals_dict_prostate = {"194 (F1)": ["Prostate pre"],
                            "194 (F2)": ["Prostate_pre"],
                            "195 (F1)": ["Prostate biop"],
                            "195 (F2)": ["Prostate pre"],
                            "196 (F1)": ["Prostate_pre"],
                            "196 (F2)": ["Prostate_pre"],
                            "199 (F1)": ["Prostate_pre"],
                            "199 (F2)": ["Prostate_pre"],
                            "198 (F2)": ["Prostate_pre", "Prostate_biop"],
                            "200 (F1)": ["Prostate_pre"],
                            "200 (F2)": ["Prostate_pre"],
                            "201 (F1)": ["Prostate_pre"],
                            "201 (F2)": ["Prostate pre"],
                            "203 (F1)": ["Prostate_pre"],
                            "203 (F2)": ["Prostate-pre"]
                            }
    
    data_removals_dict_urethra = {"194 (F1)": ["Opti Urethra"],
                            "194 (F2)": ["Opti Urethra"],
                            "195 (F1)": ["Opti Urethra", "Urethra_pre"],
                            "195 (F2)": ["Opti Urethra", "Urethra_pre"],
                            "196 (F1)": ["Opti Urethra"],
                            "196 (F2)": ["Opti Urethra"],
                            "199 (F1)": ["Opti Urethra"],
                            "199 (F2)": ["Opti Urethra"],
                            "198 (F2)": ["Opti Urethra"],
                            "200 (F1)": ["Opti Urethra"],
                            "200 (F2)": ["Opti Urethra"],
                            "201 (F1)": ["Opti Urethra"],
                            "201 (F2)": ["Opti Urethra"],
                            "203 (F1)": ["Opti Urethra"],
                            "203 (F2)": ["Opti Urethra"]
                            }
    
    data_removals_dict_rectum = {}


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








         

    ### DEFAULT MUs and SIGMAs ###

    # Prostate
    # FROM LITERATURE OF INTEROBSERVER VARIABILITY IN PROSTATE
    # "Comparison of prostate volume, shape, and contouring variability determined from preimplant magnetic resonance and transrectal ultrasound images" - Liu et al.
    # Took half of the length width height values from FIG 3.
    # Translations
    oar_default_sigma_X_list = [2.5] # default sigma in mm # 2.5 for contouring variability and 2.5 for MDA registration uncertainty, also consistent with literature
    oar_default_sigma_Y_list = [2.5] # default sigma in mm
    oar_default_sigma_Z_list = [2.5] # default sigma in mm
    oar_default_mu_X_list = [0]
    oar_default_mu_Y_list = [0]
    oar_default_mu_Z_list = [0]
    # dilations
    # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat, and p' = p + d.rhat, where rhat = p-c, where c is the non-bx structure centroid.... are mathematically equivalent, and therefore can be applied to the bx structures instead of the non bx structures!
    oar_dilations_default_sigma_XY_list = [0] # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat
    oar_dilations_default_sigma_Z_list = [0] # these are distances in mm that the points will shift towards and away from the oar centroid
    oar_dilations_default_mu_XY_list = [0]
    oar_dilations_default_mu_Z_list = [0]
    # rotations
    oar_rotations_default_sigma_X_list = [0] # pi/36 = 5 deg
    oar_rotations_default_sigma_Y_list = [0]
    oar_rotations_default_sigma_Z_list = [0]
    oar_rotations_default_mu_X_list = [0]
    oar_rotations_default_mu_Y_list = [0]
    oar_rotations_default_mu_Z_list = [0]

    # Biopsy
    # THIS SHOULD COME FROM MEAN MDA IN US TO US, THE OTHER COMPONENT COMES FROM MEAN VARIATION IN BIOPSY CENTROIDS AND IS CALCULATED IN PREPROCESSING
    # Translations
    biopsy_default_sigma_X_list = [2.5] # default sigma in mm  2.5 for MDA registration uncertainty, also consistent with literature
    biopsy_default_sigma_Y_list = [2.5] # default sigma in mm
    biopsy_default_sigma_Z_list = [2.5] # default sigma in mm
    biopsy_default_mu_X_list = [0]
    biopsy_default_mu_Y_list = [0]
    biopsy_default_mu_Z_list = [0]
    # dilations (UNIFORM) # note that these are distances (d) so 0 will impose a shift of 0 from its original position
    biopsy_dilations_default_sigma_XY_list = [0] # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat. these are distances in mm that the points will shift towards and away from the biopsy centroid line in the perpendicular radial direction
    biopsy_dilations_default_sigma_Z_list = [0] # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat. these are distances in mm that the points will shift towards and away from the biopsy centroid line in the parallel axial direction
    biopsy_dilations_default_mu_XY_list = [0]
    biopsy_dilations_default_mu_Z_list = [0]
    # rotations
    biopsy_rotations_default_sigma_X_list = [0] # pi/36 = 5 deg
    biopsy_rotations_default_sigma_Y_list = [0]
    biopsy_rotations_default_sigma_Z_list = [0]
    biopsy_rotations_default_mu_X_list = [0]
    biopsy_rotations_default_mu_Y_list = [0]
    biopsy_rotations_default_mu_Z_list = [0]


    # DILs
    # CALCULATE FROM MEAN MDA BETWEEN MRI/US
    # Translations
    dil_default_sigma_X_list = [2.5,2.5,2.5] # default sigma in mm # 2.5 for contouring variability and 2.5 for MDA registration uncertainty, also consistent with literature
    dil_default_sigma_Y_list = [2.5,2.5,2.5] # default sigma in mm
    dil_default_sigma_Z_list = [2.5,2.5,2.5] # default sigma in mm
    dil_default_mu_X_list = [0]
    dil_default_mu_Y_list = [0]
    dil_default_mu_Z_list = [0]
    # dilations
    # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat, and p' = p + d.rhat, where rhat = p-c, where c is the non-bx structure centroid.... are mathematically equivalent, and therefore can be applied to the bx structures instead of the non bx structures!
    dil_dilations_default_sigma_XY_list = [0] # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat
    dil_dilations_default_sigma_Z_list = [0] # these are distances in mm that the points will shift towards and away from the oar centroid
    dil_dilations_default_mu_XY_list = [0]
    dil_dilations_default_mu_Z_list = [0]
    # rotations
    dil_rotations_default_sigma_X_list = [0] # pi/36 = 5 deg
    dil_rotations_default_sigma_Y_list = [0]
    dil_rotations_default_sigma_Z_list = [0]
    dil_rotations_default_mu_X_list = [0]
    dil_rotations_default_mu_Y_list = [0]
    dil_rotations_default_mu_Z_list = [0]


    # Urethras
    # Translations
    urethra_default_sigma_X_list = [2.5] # default sigma in mm # 2.5 for contouring variability and 2.5 for MDA registration uncertainty, also consistent with literature
    urethra_default_sigma_Y_list = [2.5] # default sigma in mm
    urethra_default_sigma_Z_list = [2.5] # default sigma in mm
    urethra_default_mu_X_list = [0]
    urethra_default_mu_Y_list = [0]
    urethra_default_mu_Z_list = [0]
    # dilations
    # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat, and p' = p + d.rhat, where rhat = p-c, where c is the non-bx structure centroid.... are mathematically equivalent, and therefore can be applied to the bx structures instead of the non bx structures!
    urethra_dilations_default_sigma_XY_list = [0] # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat
    urethra_dilations_default_sigma_Z_list = [0] # these are distances in mm that the points will shift towards and away from the oar centroid
    urethra_dilations_default_mu_XY_list = [0]
    urethra_dilations_default_mu_Z_list = [0]
    # rotations
    urethra_rotations_default_sigma_X_list = [0] # pi/36 = 5 deg
    urethra_rotations_default_sigma_Y_list = [0]
    urethra_rotations_default_sigma_Z_list = [0]
    urethra_rotations_default_mu_X_list = [0]
    urethra_rotations_default_mu_Y_list = [0]
    urethra_rotations_default_mu_Z_list = [0]



    # Rectums
    # Translations
    rectum_default_sigma_X_list = [2.5] # default sigma in mm # 2.5 for contouring variability and 2.5 for MDA registration uncertainty, also consistent with literature
    rectum_default_sigma_Y_list = [2.5] # default sigma in mm
    rectum_default_sigma_Z_list = [2.5] # default sigma in mm
    rectum_default_mu_X_list = [0]
    rectum_default_mu_Y_list = [0]
    rectum_default_mu_Z_list = [0]
    # dilations
    # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat, and p' = p + d.rhat, where rhat = p-c, where c is the non-bx structure centroid.... are mathematically equivalent, and therefore can be applied to the bx structures instead of the non bx structures!
    rectum_dilations_default_sigma_XY_list = [0] # these are used to compute the uniform expansion distances (d). in other words b' = b-d.rhat
    rectum_dilations_default_sigma_Z_list = [0] # these are distances in mm that the points will shift towards and away from the oar centroid
    rectum_dilations_default_mu_XY_list = [0]
    rectum_dilations_default_mu_Z_list = [0]
    # rotations
    rectum_rotations_default_sigma_X_list = [0] # pi/36 = 5 deg
    rectum_rotations_default_sigma_Y_list = [0]
    rectum_rotations_default_sigma_Z_list = [0]
    rectum_rotations_default_mu_X_list = [0]
    rectum_rotations_default_mu_Y_list = [0]
    rectum_rotations_default_mu_Z_list = [0]

    
    use_added_in_quad_errors_as = 'two sigma' # can be 'sigma' or 'two sigma', 'two sigma' will provide tighter uncertainty clouds 
    biopsy_variation_uncertainty_setting = "Per biopsy mean" # Can be "Per biopsy max", "Per biopsy mean" or "Default only" .... See function (uncertainty_file_preper_by_struct_type_dataframe_NEW) defined in uncertainty_file_writer
    # "Per biopsy max" = will automatically alter uncertainty file to include the max variation of the biopsy contours for each biopsy seperately in the sigma value for the biopsy uncertainty
    # "Per biopsy mean" = will automatically alter uncertainty file to include the mean variation of the biopsy contours for each biopsy seperately in the sigma value for the biopsy uncertainty
    # "Default only" = will only use the values provided by the biospy_default_list, presumably this would account only for registration uncertainty
    non_biopsy_variation_uncertainty_setting = "Default only" # At the moment, only "Default only" is supported
    
    
    uncertainty_folder_name = 'Uncertainty data'
    uncertainty_file_name = "uncertainties_file_auto_generated"
    uncertainty_file_extension = ".csv"













    spinner_type = 'moon' # other decent ones are 'point' and 'line' or 'line2'
    rich_live_display_bool = True # [FIRST_PASS_CONFIG] If False, disables the Rich live screen and falls back to plain console status/prompt output.
    output_folder_name = 'Output data'
    preprocessed_data_folder_name = 'Preprocessed data'
    preprocessed_master_structure_ref_dict_for_export_name = 'master_structure_reference_dict'
    preprocessed_master_structure_info_dict_for_export_name = 'master_structure_info_dict'
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
    simulated_biopsy_planning_radius_mm = biopsy_radius # Keep planned sim-biopsy geometry aligned with finalized biopsy geometry and optimizer-v2 sampling.
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
    bx_sample_pts_lattice_spacing = 1
    num_MC_containment_simulations_input = 10000
    keep_light_containment_and_distances_to_relative_structures_dataframe_bool = True # This option specifies whether we keep the dataframe that gives all trial information between containment and distance between biopsy and relative structures. Note that each biopsy dataframe is about 100 MB
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
    cuml_NN_algo = 'brute' # not sure what the other options are for cuml, using brute because I want absolute accuracy
    nn_search_end_cap_grid_factor = 0.1
    svg_image_scale = 1 # setting this value to something not equal to 1 produces misaligned plots with multiple traces!
    svg_image_height = 1080
    svg_image_width = 1920
    optimizer_v2_initial_trial_prefix = 16 # minimum shared trial prefix used before the first adaptive prune round
    optimizer_v2_trial_block_size = 16 # minimum appended shared trial block per adaptive prune round
    optimizer_v2_max_total_trials = 256 # hard optimizer ceiling before final winner-resolution rescoring
    optimizer_v2_max_test_structures_per_call = None # Fixed kernel-call structure budget override. Leave as None to auto-calibrate once per optimizer-v2 run.
    optimizer_v2_fallback_max_test_structures_per_call = 4000000 # Static carry-forward structure budget derived from the last successful ~4.4M calibration on this machine; used when auto-calibration is disabled or if calibration fails.
    optimizer_v2_auto_calibrate_max_test_structures_per_call = True # When True and no fixed override is supplied, estimate a safe package-level call budget once against the run's worst-case geometry.
    optimizer_v2_verify_calibrated_max_test_structures_per_call = False # Applies only to the auto-calibration path: False = use the estimated budget directly; True = run the expensive real-call verification loop.
    optimizer_v2_mean_pd_stage_prune_std_dev_threshold = 1.0 # Adaptive mean_pd rounds require a non-None threshold; tune this to prune more or less aggressively.
    optimizer_v2_search_config = build_optimizer_v2_adaptive_block_search_config(
        initial_trial_prefix=optimizer_v2_initial_trial_prefix,
        trial_block_size=optimizer_v2_trial_block_size,
        max_total_trials=optimizer_v2_max_total_trials,
        mean_pd_stage_prune_std_dev_threshold=optimizer_v2_mean_pd_stage_prune_std_dev_threshold,
        max_test_structures_per_call=optimizer_v2_max_test_structures_per_call,
    )
    optimizer_v2_max_candidates_per_chunk = None # Optimizer-level outer candidate chunk override. Leave as None to derive it dynamically from the calibrated structure budget; set a positive int to force a fixed outer chunk size without changing the CUDA containment module boundary.
    optimizer_v2_validate_nearest_z_helper_against_ver5_bool = False # If True, validate the active grouped nearest-z helper against ver5 during optimizer-v2 scoring and log the exact-match result.
    optimizer_v2_benchmark_isolated_winner_validation_bool = True # If True, rerun the final winner once more in isolation at the downstream-comparable trial count and log a direct benchmark. This adds one extra winner-validation-like pass per structure.
    optimizer_v2_render_stage_boundary_candidate_clouds_bool = False # HERE # Opens one stage-switchable scene per v2 biopsy. Set False to render none.
    optimizer_v2_render_stage_names = None # None = render every adaptive prune round in order.
    optimizer_v2_render_backend = "both" # open3d = multistage debug viewer, plotly = one scientific figure per rendered stage, both = run both backends.
    optimizer_v2_render_plotly_export_bool = False # HERE # If True, export publication-oriented Plotly vector figures for the selected optimizer-v2 scenes.
    optimizer_v2_render_plotly_export_formats = ("svg", "pdf")
    optimizer_v2_render_plotly_export_width = svg_image_width
    optimizer_v2_render_plotly_export_height = svg_image_height
    optimizer_v2_render_plotly_export_scale = svg_image_scale
    optimizer_v2_render_plotly_export_camera_eye = (1.45, -1.45, 2.25)
    optimizer_v2_render_plotly_export_camera_center = (0.0, 0.0, 0.0)
    optimizer_v2_render_plotly_export_camera_up = (0.0, 0.0, 1.0)
    optimizer_v2_render_dialog_timeout_seconds = None # None waits indefinitely; set a positive number to auto-continue unattended render dialogs.
    optimizer_v2_render_dialog_timeout_extend_seconds = 300.0 # Clicking More time adds this many seconds to the current render-dialog timeout.
    optimizer_v2_render_winner_containment_debug_bool = False # HERE # If True, rerun the winning candidate with debug-localized points and render success/failure stochastic clouds against the target.
    optimizer_v2_render_winner_containment_backend = "both" # open3d, plotly, both, or none for export-only.
    optimizer_v2_render_include_target_points_bool = False # If False, omit the raw DIL point cloud and rely on contour-style target layers instead.
    optimizer_v2_render_include_target_surface_bool = True # If True, show the target DIL contour surface layer in addition to the target-point cloud layer.
    optimizer_v2_render_patient_whitelist = None # None = all patients, () = none, non-empty tuple = exact patient filter.
    optimizer_v2_render_roi_whitelist = None # None = all ROIs, () = none, non-empty tuple = case-insensitive substring filter.
    optimizer_v2_render_layer_style_by_name = {
        "stage_input_candidates": {"color": np.array([0.88, 0.53, 0.10]), "marker_size": 2.0, "opacity": 0.28},
        "stage_survivors": {"color": np.array([0.14, 0.68, 0.24]), "marker_size": 3.2, "opacity": 0.88},
        "target_points": {"color": np.array([0.33, 0.63, 0.33]), "marker_size": 0.7, "opacity": 0.10},
        "target_structure_centroid": {"marker_size": 8.0, "opacity": 1.0},
        "nominal_biopsy_centroid": {"color": np.array([0.85, 0.20, 0.20]), "marker_size": 7.0, "opacity": 1.0},
        "operational_winner": {"color": np.array([0.86, 0.12, 0.68]), "marker_size": 8.0, "opacity": 1.0},
        "planned_sampled_points": {"marker_size": 1.8, "opacity": 0.40},
        "planned_core_structure": {"line_width": 5.0, "opacity": 0.98},
        "planned_centroid_line": {"line_width": 6.0, "opacity": 1.0},
        "target_structure_surface": {"line_width": 4.8, "opacity": 0.96},
        "prostate_structure": {"line_width": 4.0, "opacity": 0.90},
        "urethra_structure": {"line_width": 4.5, "opacity": 1.0},
        "rectum_structure": {"line_width": 3.8, "opacity": 0.88},
    }
    num_stochastic_targeting_transform_samples_input = 0 # Placeholder budget for a future stochastic-targeting stage before simulated-biopsy planning; currently only used when sizing shared transform precompute.
    transform_generation_random_seed = 51
    optimizer_v1_random_seed = 51

    # custom point containment algorithm options
    generate_cuda_log_files_MC_containment_sim = False
    generate_cuda_log_files_volume_calculation = False
    generate_cuda_log_files_structure_dimension_calculation = False
    generate_cuda_log_files_biopsy_optimizer = False
    include_edges_in_log_files = False

    ### Kernel selection:
    """
    1. The type of kernel to use. The default is "one_to_one_pip_kernel_advanced". 
    2. The other option is "one_to_one_pip_kernel_advanced_reparameterized_version" which is a version of that kernel that ALSO uses the reparameterized version of the mathematics which should in theory be more robust to regenerating rays. 
    3. (MOST ADVANCED VERSION) The other is "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized" which implements much better practices of gpu memory and performance optimization by not calculating poly_points at all, and passing pointers to indices instead to the kernel.
    """
    custom_cuda_kernel_type = "one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized" 
    constant_z_slice_polygons_handler_option = 'auto-close-if-open' # Can be 'auto-close-if-open' or 'close-all' or None
    remove_consecutive_duplicate_points_in_polygons = True

    num_dose_NN_to_show_for_animation_plotting = 100
    num_bootstraps_for_regression_plots_input = 15
    pio.templates.default = "plotly_white"
    NPKR_bandwidth = 0.5
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
    optimal_normal_dist_option = 'dil dimension driven' # can be 'biopsy_and_dil_sigmas' or 'dil dimension driven', note that the biopsy_and_dil_sigmas option adds all sigmas in quadrature and then uses this value as TWO sigma. Note that the dil deimnsion driven option uses the dimension of the respective dil at the position of the dil centroid in each direction as TWO sigma
    # these multipliers provide a lengthening or stretching of the normal dist to bias a certain dimension as relatively more important 
    bias_LR_multiplier = 1
    bias_AP_multiplier = 1
    bias_SI_multiplier = 1.5 
    # for guidance maps 
    number_of_optimal_template_holes_to_consider_for_guidance_maps_firing_depth_recommendation = 3 # number of optimal template holes to consider for guidance maps firing depth recommendation
    render_guidance_maps_after_simulated_core_finalization = False
    guidance_map_plot_name = "guidance maps"
    guidance_map_output_dir_name = "Guidance maps"
    guidance_map_save_formats = ("svg", "pdf", "html")
    guidance_map_image_width = 1300
    guidance_map_image_height = 1300
    show_titles_for_guidance_maps = False
    # Guidance-map plotting rank policy:
    #   - int (e.g., 1 or 2): render that rank only
    #   - list of ints (e.g., [1, 2, 3]): attempt each in order
    #   - "all": render all available ranks for each DIL
    candidate_plot_ranks_behavior = 'all'
    # Validation CSV export toggle for guidance-map precomputed inputs/contracts/selection manifest.
    validate_firing_df_builder_behavior = False # this should be turned on for guidance map building in the future, im turning it off for now because it takes a long time
    validate_phase3b_in_memory_patient_stitching_bool = True
    write_phase3b_in_memory_stitched_tables_bool = True
    write_phase3c_patient_fragment_output_surface_bool = True
    write_phase3c_stitched_final_artifacts_bool = True
    patient_runner_validation_mode = PatientRunnerMainValidationMode.SHADOW_OUTPUT.value
    patient_runner_validation_patient_uids = ()
    patient_runner_validation_final_table_names = ()
    patient_runner_validation_source_table_names = ()
    patient_runner_validation_write_outputs_bool = True
    patient_runner_validation_write_assembled_tables_bool = True
    # Strict mode policy:
    #   - True: fail fast on missing/invalid rank data (raises)
    #   - False: skip problematic ranks, keep run alive, and log details in validation manifest/notes
    strict_precomputed_guidance_behavior = False
    # If False, Euler-angle annotation box is hidden on the map; Euler values remain in compact tables.
    show_euler_annotation_box_behavior = False

    # for simulated biopsies
    centroid_dil_sim_key = 'Centroid DIL'
    optimal_dil_sim_key = 'Optimal DIL'
    target_dil_v2_sim_key = 'Target DIL v2'
    bx_sim_locations_dict = {centroid_dil_sim_key:
                                                            {"Create": True,
                                                            "Relative to struct type": dil_ref,
                                                            "Transport family": "centroid",
                                                            "Identifier string": 'sim_centroid_dil'}
                                                            ,   
                                                        optimal_dil_sim_key:
                                                            {"Create": True,
                                                            "Relative to struct type": dil_ref,
                                                            "Transport family": "optimal",
                                                            "Identifier string": 'sim_optimal_dil'}
                                                            ,
                                                        target_dil_v2_sim_key:
                                                            {"Create": True,
                                                            "Relative to struct type": dil_ref,
                                                            "Transport family": "identity",
                                                            "Identifier string": 'sim_target_dil_v2'}
                                                        }
    simulated_biopsy_fraction_numbers_to_create = 'all'   # [FIRST_PASS_CONFIG] use [2] for legacy F2-only behavior
    simulated_biopsy_length_method = 'match real'   # [FIRST_PASS_CONFIG] can be 'full' or 'match real'. Cohort-mean length modes were removed for patient-runner compatibility.
                                                    # 'match real' uses a matched real biopsy length, then same-patient/same-DIL mean if available, then the full needle compartment length.
    color_discrete_map_by_sim_type = {'Real': 'rgba(0, 92, 171, 1)', centroid_dil_sim_key: 'rgba(227, 27, 35,1)', optimal_dil_sim_key: 'rgba(0, 0, 0,1)', target_dil_v2_sim_key: 'rgba(26, 71, 42, 1)'}
    biopsy_pcd_colors_dict = {'Real': np.array([0.5, 0.0, 0.5]), centroid_dil_sim_key: np.array([1.0, 0.55, 0.0]), optimal_dil_sim_key: np.array([0.0, 0.8, 0.6]), target_dil_v2_sim_key: np.array([0.1, 0.65, 0.2])} # real: purple, centroid: deep orange, optimal: light teal, target-v2: deep green

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

    # patient sample cohort analyzer
    box_plot_points_option = 'outliers'
    notch_option = False
    boxmean_option = True # can be 'sd' or True

    
    
    
    
    ### PLOTS TO SHOW:

    # Preprocessing
    demonstrate_volume_calculation_correctness_bool_1 = False # Volume ---- NEW CUSTOM CONTAINMENT ALGO: shows the volume calculation from PIP test
    plot_volume_calculation_containment_result_bool_1_old = False # Volume ---- OLD CUSPATIAL CONTAINMENT ALGO: shows the volume calculation from PIP test
    demonstrate_structure_dimension_calculation_correctness_bool_1 = False # Dimension ---- NEW CUSTOM CONTAINMENT ALGO: shows the dimension calculation from PIP test
    demonstrate_structure_dimension_calculation_correctness_bool_1_old = False # Dimension ---- OLD CUSPATIAL CONTAINMENT ALGO: shows the dimension calculation from PIP test


    # Transformations
    inspect_self_biopsy_dilate_bool = False # per trial basis
    inspect_self_biopsy_dilate_and_rotate_bool = False # per trial basis
    inspect_self_biopsy_dilate_and_rotate_and_translate_bool = False # per trial basis
    inspect_relative_structure_rotate_and_shift_number = 0 # per trial basis, if 0 will not show any plots, make sure this number is less than the num_containment_sims value !
    show_non_bx_relative_structure_z_dilation_bool = False # per trial basis
    show_non_bx_relative_structure_xy_dilation_bool = False # per trial basis

    # Dosimetry
    show_NN_dose_demonstration_plots = False # this shows one trial at a time!!!
    show_3d_dose_renderings = False
    show_3d_dose_renderings_thresholded = False
    show_NN_dose_demonstration_plots_all_trials_at_once = False # nice because shows all trials at once

    # Tissue class and structure distances
    show_num_containment_demonstration_plots = 0 # this shows one trial at a time!!!
    containment_results_structure_types_to_show_per_trial = [oar_ref, dil_ref] # can be any combination of the structure references
    plot_cupy_containment_distribution_results = False # nice because it shows all trials at once
    show_num_nearest_neighbour_surface_boundary_demonstration = 0 # must be an integer, 0 means show none, you see one trial at a time
    show_num_relative_structure_centroid_demonstration = 0 # must be an integer, 0 means show none, you see one trial at a time
    check_if_end_caps_filled_proper_NN_num = 0

    # DIL biopsy optimization
    demonstrate_dil_optimization_points_inside_correctness_bool_1 = False # shows the containment results for the generated lattice points that is passed to the optimizer function
    demonstrate_dil_optimization_points_inside_correctness_bool_2 = False # shows the containment results for the generated lattice inside the optiomizer function, which is only caleld if you didnt pass the optimizer function a lattice. We are currently passing it a lattice.
    demonstrate_dil_optimization_points_inside_correctness_num_3 = 0 # 0, means off! Shows the containment results for all normal distritution generated points centered at each test point lattice position, shows random 5 trials
    plot_each_normal_dist_containment_result_bool = False
    plot_optimization_point_lattice_bool = False
    show_optimization_point_bool = False
    display_optimization_contour_plots_bool = False

    # MRs
    show_3d_mr_adc_renderings = False
    show_3d_mr_adc_renderings_thresholded = False
    show_NN_mr_adc_demonstration_plots = False # this shows one trial at a time!!
    show_NN_mr_adc_demonstration_plots_all_trials_at_once = False # nice because shows all trials at once
    demonstrate_mr_adc_pcd_containment_correctness_bool = False # This shows the containment results for the MR ADC point cloud within prostate, which is generated from the MR ADC image
    demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool = False # This shows the containment results for the MR ADC point cloud within prostate ONLY, ie urethra , DIL and rectum points have been removed
    
    # Combined
    show_processed_3d_datasets_renderings = False
    show_processed_3d_datasets_renderings_plotly_dict = {"Plot": False, # If false then the rest of the options are irrelevant
                                                         "SS Scatter": False,
                                                         "SS Contour": True,
                                                         "Dosimetric render mode": "volume", # can be "volume" or "scatter"
                                                         "Dosimetric dose log scale": True, # If false then its linear
                                                         "mr render mode": "volume", # can be "volume" or "scatter"
                                                         "mr log scale": False, # If false then its linear
                                                         }


    # Misc
    show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot = False
    plot_uniform_shifts_to_check_plotly = False # if this is true, will produce many plots if num_simulations is high!
    plot_translation_vectors_pointclouds = False
    plot_shifted_biopsies = False
    display_curvature_bool = False
    display_structure_surface_mesh_bool = False
    plot_binary_mask_bool = False
    plot_guidance_map_transducer_plane_open3d_structure_set_complete_demonstration_bool = False
    show_equivalent_ellipsoid_from_pca_bool = False
    display_pca_fit_variation_for_biopsies_bool = False
    validate_non_biopsy_structure_preprocessing_equivalence_bool = True # Runs the modular helper on the live structure, shadows it with the legacy inline path for validation, then restores the modular outputs. Intended for focused validation only.

    ###







    ### Legacy plotting controls retained only for local simulated-biopsy playback.
    
    plot_immediately_after_simulation = True
    # other parameters
    modify_generated_uncertainty_template = False # if True, the algorithm wont be able to run from start to finish without an interupt, allowing one to modify the uncertainty file
    write_containment_to_file_ans = True # If True, this generates and saves to file a csv file of the containment simulation
    write_dose_to_file_ans = True # If True, this generates and saves to file a csv file of the dose simulation
    export_pickled_preprocessed_data = False # If True, this exports a pickled version of master_structure_reference_dict and master_structure_info_dict
    skip_preprocessing = False # If True, you will be asked to specify the locations of master_structure_info_dict and master_structure_reference_dict
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
    prostate_tissue_label = 'Prostatic'
    rectal_tissue_label = 'Rectal'
    urethral_tissue_label = 'Urethral'

    # Tissue volume threshold operator dictionary
    # This is a dictionary that contains the operator to use for the volume thresholding.
    tissue_volume_operator_dictionary = {cancer_tissue_label: 'greater',
                                         prostate_tissue_label: 'greater',
                                         rectal_tissue_label: 'less',
                                         urethral_tissue_label: 'less',
                                         default_exterior_tissue: 'less'
                                         }


    # non-user changeable variables, but need to be initiatied:
    
    # DO NOT CHANGE THE ORDER OF THE KEYS IN THE BELOW DICTIONARY!!!! 
    structs_referenced_dict = { bx_ref: {"Contour names": biopsy_contour_names,
                                        "Default mu X": biopsy_default_mu_X_list,
                                        "Default mu Y": biopsy_default_mu_Y_list,
                                        "Default mu Z": biopsy_default_mu_Z_list,  
                                        "Default sigma X": biopsy_default_sigma_X_list,
                                        "Default sigma Y": biopsy_default_sigma_Y_list,
                                        "Default sigma Z": biopsy_default_sigma_Z_list,
                                        "Dilations mu (xy)": biopsy_dilations_default_mu_XY_list,
                                        "Dilations mu (z)": biopsy_dilations_default_mu_Z_list,
                                        "Dilations sigma (xy)": biopsy_dilations_default_sigma_XY_list,
                                        "Dilations sigma (z)": biopsy_dilations_default_sigma_Z_list,
                                        "Rotations mu X": biopsy_rotations_default_mu_X_list,
                                        "Rotations mu Y": biopsy_rotations_default_mu_Y_list,
                                        "Rotations mu Z": biopsy_rotations_default_mu_Z_list,
                                        "Rotations sigma X": biopsy_rotations_default_sigma_X_list,
                                        "Rotations sigma Y": biopsy_rotations_default_sigma_Y_list,
                                        "Rotations sigma Z": biopsy_rotations_default_sigma_Z_list,
                                        'Test tissue class': None, # should always be None
                                        'Tissue heirarchy': None, # should always be None
                                        'Tissue class name': None, # Not used for anything as of yet..
                                        'PCD color dict': biopsy_pcd_colors_dict
                                        }, 
                                oar_ref: {"Contour names": oaroi_contour_names,
                                          "Default mu X": oar_default_mu_X_list,
                                          "Default mu Y": oar_default_mu_Y_list,
                                          "Default mu Z": oar_default_mu_Z_list, 
                                          "Default sigma X": oar_default_sigma_X_list,
                                          "Default sigma Y": oar_default_sigma_Y_list,
                                          "Default sigma Z": oar_default_sigma_Z_list,
                                          "Dilations mu (xy)": oar_dilations_default_mu_XY_list,
                                          "Dilations mu (z)": oar_dilations_default_mu_Z_list,
                                          "Dilations sigma (xy)": oar_dilations_default_sigma_XY_list,
                                          "Dilations sigma (z)": oar_dilations_default_sigma_Z_list,
                                          "Rotations mu X": oar_rotations_default_mu_X_list,
                                          "Rotations mu Y": oar_rotations_default_mu_Y_list,
                                          "Rotations mu Z": oar_rotations_default_mu_Z_list,
                                          "Rotations sigma X": oar_rotations_default_sigma_X_list,
                                          "Rotations sigma Y": oar_rotations_default_sigma_Y_list,
                                          "Rotations sigma Z": oar_rotations_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 3,
                                          'Tissue class name': prostate_tissue_label, 
                                          'PCD color': np.array([0.86, 0.08, 0.24]) # crimson
                                          }, 
                                dil_ref: {"Contour names": dil_contour_names,
                                          "Default mu X": dil_default_mu_X_list,
                                          "Default mu Y": dil_default_mu_Y_list,
                                          "Default mu Z": dil_default_mu_Z_list, 
                                          "Default sigma X": dil_default_sigma_X_list,
                                          "Default sigma Y": dil_default_sigma_Y_list,
                                          "Default sigma Z": dil_default_sigma_Z_list,
                                          "Dilations mu (xy)": dil_dilations_default_mu_XY_list,
                                          "Dilations mu (z)": dil_dilations_default_mu_Z_list,
                                          "Dilations sigma (xy)": dil_dilations_default_sigma_XY_list,
                                          "Dilations sigma (z)": dil_dilations_default_sigma_Z_list,
                                          "Rotations mu X": dil_rotations_default_mu_X_list,
                                          "Rotations mu Y": dil_rotations_default_mu_Y_list,
                                          "Rotations mu Z": dil_rotations_default_mu_Z_list,
                                          "Rotations sigma X": dil_rotations_default_sigma_X_list,
                                          "Rotations sigma Y": dil_rotations_default_sigma_Y_list,
                                          "Rotations sigma Z": dil_rotations_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 0,
                                          'Tissue class name': cancer_tissue_label,
                                          'PCD color': np.array([0.13, 0.55, 0.13]) # forest green
                                          },
                                rectum_ref_key: {"Contour names": rectum_contour_names,
                                          "Default mu X": rectum_default_mu_X_list,
                                          "Default mu Y": rectum_default_mu_Y_list,
                                          "Default mu Z": rectum_default_mu_Z_list, 
                                          "Default sigma X": rectum_default_sigma_X_list,
                                          "Default sigma Y": rectum_default_sigma_Y_list,
                                          "Default sigma Z": rectum_default_sigma_Z_list,
                                          "Dilations mu (xy)": rectum_dilations_default_mu_XY_list,
                                          "Dilations mu (z)": rectum_dilations_default_mu_Z_list,
                                          "Dilations sigma (xy)": rectum_dilations_default_sigma_XY_list,
                                          "Dilations sigma (z)": rectum_dilations_default_sigma_Z_list,
                                          "Rotations mu X": rectum_rotations_default_mu_X_list,
                                          "Rotations mu Y": rectum_rotations_default_mu_Y_list,
                                          "Rotations mu Z": rectum_rotations_default_mu_Z_list,
                                          "Rotations sigma X": rectum_rotations_default_sigma_X_list,
                                          "Rotations sigma Y": rectum_rotations_default_sigma_Y_list,
                                          "Rotations sigma Z": rectum_rotations_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 2,
                                          'Tissue class name': rectal_tissue_label,
                                          'PCD color': np.array([1.0, 0.84, 0.0]) # gold
                                          },
                                urethra_ref_key: {"Contour names": urethra_contour_names,
                                          "Default mu X": urethra_default_mu_X_list,
                                          "Default mu Y": urethra_default_mu_Y_list,
                                          "Default mu Z": urethra_default_mu_Z_list, 
                                          "Default sigma X": urethra_default_sigma_X_list,
                                          "Default sigma Y": urethra_default_sigma_Y_list,
                                          "Default sigma Z": urethra_default_sigma_Z_list,
                                          "Dilations mu (xy)": urethra_dilations_default_mu_XY_list,
                                          "Dilations mu (z)": urethra_dilations_default_mu_Z_list,
                                          "Dilations sigma (xy)": urethra_dilations_default_sigma_XY_list,
                                          "Dilations sigma (z)": urethra_dilations_default_sigma_Z_list,
                                          "Rotations mu X": urethra_rotations_default_mu_X_list,
                                          "Rotations mu Y": urethra_rotations_default_mu_Y_list,
                                          "Rotations mu Z": urethra_rotations_default_mu_Z_list,
                                          "Rotations sigma X": urethra_rotations_default_sigma_X_list,
                                          "Rotations sigma Y": urethra_rotations_default_sigma_Y_list,
                                          "Rotations sigma Z": urethra_rotations_default_sigma_Z_list,
                                          'Test tissue class': True,
                                          'Tissue heirarchy': 1,
                                          'Tissue class name': urethral_tissue_label,
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
    pipeline_config = PipelineConfig(
        ui=RuntimeUIConfig(
            spinner_type=spinner_type,
            rich_live_display_bool=rich_live_display_bool,
        ),
        artifacts=ArtifactConfig(
            output_folder_name=output_folder_name,
            preprocessed_data_folder_name=preprocessed_data_folder_name,
            preprocessed_reference_dict_filename=preprocessed_master_structure_ref_dict_for_export_name,
            preprocessed_info_dict_filename=preprocessed_master_structure_info_dict_for_export_name,
            export_pickled_preprocessed_data=export_pickled_preprocessed_data,
            skip_preprocessing=skip_preprocessing,
        ),
        preprocessing=PreprocessingConfig(
            interp_inter_slice_dist=interp_inter_slice_dist,
            interp_intra_slice_dist=interp_intra_slice_dist,
            interp_dist_caps=interp_dist_caps,
            radius_for_normals_estimation=radius_for_normals_estimation,
            max_nn_for_normals_estimation=max_nn_for_normals_estimation,
            voxel_size_for_structure_volume_calc_non_bx=voxel_size_for_structure_volume_calc_non_bx,
            voxel_size_for_structure_dimension_calc=voxel_size_for_structure_dimension_calc,
            factor_for_voxel_size=factor_for_voxel_size,
            cupy_array_upper_limit_nxn_size_input=cupy_array_upper_limit_NxN_size_input,
            nearest_zslice_vals_and_indices_cupy_generic_max_size=(
                nearest_zslice_vals_and_indices_cupy_generic_max_size
            ),
            generate_cuda_log_files_volume_calculation=generate_cuda_log_files_volume_calculation,
            constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
            remove_consecutive_duplicate_points_in_polygons=(
                remove_consecutive_duplicate_points_in_polygons
            ),
            include_edges_in_log_files=include_edges_in_log_files,
            custom_cuda_kernel_type=custom_cuda_kernel_type,
            demonstrate_volume_calculation_correctness_bool_1=(
                demonstrate_volume_calculation_correctness_bool_1
            ),
            plot_volume_calculation_containment_result_bool_1_old=(
                plot_volume_calculation_containment_result_bool_1_old
            ),
            plot_binary_mask_bool=plot_binary_mask_bool,
            generate_cuda_log_files_structure_dimension_calculation=(
                generate_cuda_log_files_structure_dimension_calculation
            ),
            demonstrate_structure_dimension_calculation_correctness_bool_1=(
                demonstrate_structure_dimension_calculation_correctness_bool_1
            ),
            demonstrate_structure_dimension_calculation_correctness_bool_1_old=(
                demonstrate_structure_dimension_calculation_correctness_bool_1_old
            ),
            demonstrate_mr_adc_pcd_containment_correctness_bool=(
                demonstrate_mr_adc_pcd_containment_correctness_bool
            ),
            display_structure_surface_mesh_bool=display_structure_surface_mesh_bool,
            show_equivalent_ellipsoid_from_pca_bool=show_equivalent_ellipsoid_from_pca_bool,
        ),
        replay=RuntimeReplayConfig(
            lower_bound_dose_value=lower_bound_dose_value,
            lower_bound_dose_gradient_value=lower_bound_dose_gradient_value,
            lower_bound_mr_adc_value=lower_bound_mr_adc_value,
            upper_bound_mr_adc_value=upper_bound_mr_adc_value,
            color_flattening_deg_mr=color_flattening_deg_MR,
        ),
        guidance_maps=GuidanceMapConfig(
            planning_config=GuidanceMapPlanningConfig(
                candidate_holes_k=number_of_optimal_template_holes_to_consider_for_guidance_maps_firing_depth_recommendation,
                candidate_axis_line_length_mm=1000,
            ),
            render_config=GuidanceMapRenderConfig(
                enabled=render_guidance_maps_after_simulated_core_finalization,
                plot_name=guidance_map_plot_name,
                output_dir_name=guidance_map_output_dir_name,
                save_formats=guidance_map_save_formats,
                image_width=guidance_map_image_width,
                image_height=guidance_map_image_height,
                image_scale=svg_image_scale,
                axis_title_font_size=24,
                axis_tick_font_size=20,
                legend_font_size=20,
                annotation_font_size=20,
                distance_annotation_font_size=20,
                fire_annotation_font_size=20,
                colorbar_tick_font_size=20,
                template_label_font_size=20,
                colorbar_title_font_size=20,
                fire_annotation_style="compact_table",
                fire_table_position="outside top center",
                draw_orientation_diagram=False,
                show_titles=show_titles_for_guidance_maps,
                show_euler_annotation_box=show_euler_annotation_box_behavior,
                candidate_plot_rank=candidate_plot_ranks_behavior,
                validate_firing_df_builder=validate_firing_df_builder_behavior,
                strict_precomputed_guidance=strict_precomputed_guidance_behavior,
            )
        ),
        optimizer=OptimizerRuntimeConfig(
            optimizer_v2_search_config=optimizer_v2_search_config,
            num_stochastic_targeting_transform_samples_input=(
                num_stochastic_targeting_transform_samples_input
            ),
        ),
        random_seeds=RandomSeedConfig(
            transform_generation_random_seed=transform_generation_random_seed,
            optimizer_v1_random_seed=optimizer_v1_random_seed,
        ),
    )
    guidance_map_planning_config = pipeline_config.guidance_maps.planning_config
    guidance_map_render_config = pipeline_config.guidance_maps.render_config
    
    # initialize perform mc sim based on other parameters
    perform_mc_dose_sim = bool(num_MC_dose_simulations_input)
    perform_mc_containment_sim = bool(num_MC_containment_simulations_input)
    perform_mc_mr_sim = bool(num_MC_MR_simulations_input)
    perform_MC_sim = perform_mc_containment_sim or perform_mc_dose_sim or perform_mc_mr_sim


    # create a dict for cohort data and dataframes
    mr_global_multi_structure_output_dataframe_str = "Global MR ADC statistics"
    mr_global_by_voxel_multi_structure_output_dataframe_str = "Global by voxel MR ADC statistics"

    master_cohort_patient_data_and_dataframes = {"Data": {},
                                                 "Dataframes": {"Uncertainties dataframe (unedited)": None,
                                                                "Uncertainties dataframe (final)": None,
                                                                "Cohort: Nearest DILs to each biopsy": None,
                                                                "Cohort: Biopsy basic spatial features dataframe": None,
                                                                "Cohort: Simulated biopsy preparation dataframe": None,
                                                                "Cohort: Guidance-map firing depth recommendations dataframe": None,
                                                                "Cohort: 3D radiomic features all OAR and DIL structures": None,
                                                                "Cohort: All MC structure shift vectors": None,
                                                                "Cohort: All MC structure transformation values": None,
                                                                "Cohort: structure specific mc results": None,
                                                                "Cohort: sum-to-one mc results": None,
                                                                "Cohort: global sum-to-one mc results": None,
                                                                #"Cohort: mutual tissue class mc results": None,
                                                                #"Cohort: tissue class global scores (tissue type)": None,
                                                                "Cohort: tissue class global scores (structure)": None,
                                                                #"Cohort: Entire point-wise binom est distribution": None,
                                                                "Cohort: Entire point-wise dose distribution": None,
                                                                "Cohort: Tissue class - distances global results": None,
                                                                "Cohort: Tissue class - distances pt-wise results": None,
                                                                "Cohort: Tissue class - distances voxel-wise results": None,
                                                                "Cohort: Per sample point prostate double sextant classification": None,
                                                                "Cohort: Per voxel prostate double sextant classification": None,
                                                                "Cohort: Simulated biopsy planned vs realized centroid variation validation": None,
                                                                "Cohort: Bx DVH metrics (generalized)": None,
                                                                "Cohort: Bx global info dataframe": None,
                                                                "Cohort: "+ mr_global_multi_structure_output_dataframe_str: None,
                                                                "Cohort: "+ mr_global_by_voxel_multi_structure_output_dataframe_str: None
                                                                }
                                                 }


    runtime_logger = None
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
        
               
        with rich_preambles.get_live_display(rich_layout,
                                             refresh_per_second = 8,
                                             screen = True,
                                             rich_live_display_bool = rich_live_display_bool) as live_display:
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
            runtime_logger = install_runtime_logger(RuntimeLogger(output_dir))
            important_info.set_runtime_logger(runtime_logger)
            runtime_logger.checkpoint(
                "run.initialization.complete",
                "Initialized runtime logging and validated base input/output directories.",
                details={
                    "data_dir": data_dir,
                    "input_dir": input_dir,
                    "output_dir": output_dir,
                    "preprocessed_data_dir": preprocessed_data_dir,
                },
            )
            runtime_logger.memory_snapshot(
                "run.initialization.complete",
                "Captured initial memory snapshot after runtime initialization.",
            )
           
            section_start_time = datetime.now() 
            runtime_logger.phase_start("section.simulations", "Starting section: Simulations.")
    
            if pipeline_config.artifacts.skip_preprocessing == False:
                runtime_logger.phase_start(
                    "input.discovery",
                    "Starting DICOM input discovery.",
                    details={
                        "input_dir": input_dir,
                        "uncertainty_dir": uncertainty_dir,
                    },
                )
                

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
                        simulate_biopsies_relative_to = sim_bx_type_dict["Relative to struct type"]
                        
                        important_info.add_text_line("Simulating "+ sim_bx_type_str+" biopsies relative to "+simulate_biopsies_relative_to+".", live_display)          
                        live_display.refresh()
                else: 
                    important_info.add_text_line("Not creating any simulated biopsies.", live_display)          
                    live_display.refresh() 

                runtime_logger.phase_end(
                    "input.discovery",
                    "Completed DICOM input discovery.",
                    details={
                        "num_dicoms": num_dicoms,
                        "num_rtstruct_patients": num_RTst_dcms_entries,
                        "num_rtdose_patients": num_RTdose_dcms_entries,
                        "num_rtplan_patients": num_RTplan_dcms_entries,
                        "num_mr_t2_patients": num_MR_T2_dcms_entries,
                        "num_mr_adc_patients": num_MR_ADC_dcms_entries,
                        "num_us_patients": num_US_dcms_entries,
                    },
                )
                
                #live_display.stop()
                # patient dictionary creation
                building_patient_dictionaries_task = indeterminate_progress_main.add_task('[red]Building patient dictionary...', total=None)
                building_patient_dictionaries_task_completed = completed_progress.add_task('[green]Building patient dictionary', total=num_RTst_dcms_entries, visible = False)
                runtime_logger.phase_start(
                    "preprocessing.structure_referencer",
                    "Building patient master dictionary.",
                    details={
                        "num_rtstruct_patients": num_RTst_dcms_entries,
                        "num_rtdose_patients": num_RTdose_dcms_entries,
                        "num_rtplan_patients": num_RTplan_dcms_entries,
                    },
                )
                master_structure_reference_dict, master_structure_info_dict = structure_referencer(data_removals_dict_bx,
                                                                                                data_removals_dict_prostate,
                                                                                                data_removals_dict_dil,
                                                                                                data_removals_dict_urethra,
                                                                                                data_removals_dict_rectum,
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
                                                                                                structs_referenced_dict,
                                                                                                dose_ref,
                                                                                                plan_ref,
                                                                                                mr_adc_ref,
                                                                                                mr_t2_ref,
                                                                                                us_ref,
                                                                                                all_ref_key,
                                                                                                mr_global_multi_structure_output_dataframe_str,
                                                                                                mr_global_by_voxel_multi_structure_output_dataframe_str,
                                                                                                bx_sim_locations_dict,
                                                                                                rectum_contour_names,
                                                                                                urethra_contour_names,
                                                                                                interp_inter_slice_dist,
                                                                                                interp_intra_slice_dist,
                                                                                                simulated_biopsy_fraction_numbers_to_create,
                                                                                                fraction_prefixes,
                                                                                                important_info,
                                                                                                live_display
                                                                                                )
                #live_display.stop()
                indeterminate_progress_main.update(building_patient_dictionaries_task, visible = False)
                completed_progress.update(building_patient_dictionaries_task_completed, advance = num_RTst_dcms_entries,visible = True)
                important_info.add_text_line("Patient master dictionary built for "+str(master_structure_info_dict["Global"]["Num cases"])+" patients.", live_display)  
                runtime_logger.phase_end(
                    "preprocessing.structure_referencer",
                    "Built patient master dictionary.",
                    details={
                        "num_cases": master_structure_info_dict["Global"]["Num cases"],
                        "num_structures": master_structure_info_dict["Global"]["Num structures"],
                    },
                )
                runtime_logger.memory_snapshot(
                    "preprocessing.structure_referencer",
                    "Captured memory snapshot after structure referencer completed.",
                )
                live_display.refresh()

                configure_transform_precompute_settings(
                    master_structure_info_dict,
                    pipeline_config.optimizer.optimizer_v2_search_config,
                    pipeline_config.optimizer.num_stochastic_targeting_transform_samples_input,
                )
                configure_runtime_random_seed_settings(
                    master_structure_info_dict,
                    pipeline_config.random_seeds.transform_generation_random_seed,
                    pipeline_config.random_seeds.optimizer_v1_random_seed,
                )

                specific_output_dir, raw_mc_output_dir = create_run_output_directories(
                    master_structure_info_dict,
                    output_dir,
                )
                runtime_logger.attach_output_dir(specific_output_dir)
                runtime_logger.checkpoint(
                    "run_output_dir.ready",
                    "Initialized specific output directory for this run.",
                    details={
                        "specific_output_dir": specific_output_dir,
                        "raw_mc_output_dir": raw_mc_output_dir,
                    },
                )
                runtime_logger.memory_snapshot(
                    "run_output_dir.ready",
                    "Captured memory snapshot after creating run output directories.",
                )

                input_manifest_result = write_input_manifest_files(
                    output_dir=specific_output_dir,
                    dicom_paths=dicom_paths_list,
                    rtstruct_dcms_dict=RTst_dcms_dict,
                    rtdose_dcms_dict=RTdose_dcms_dict,
                    rtplan_dcms_dict=RTplan_dcms_dict,
                    us_dcms_dict=US_dcms_dict,
                    mr_t2_dcms_dict=MR_T2_dcms_dict,
                    mr_adc_dcms_dict=MR_ADC_dcms_dict,
                    fraction_prefixes=fraction_prefixes,
                    runtime_logger=runtime_logger,
                )
                important_info.add_text_line(
                    "Input manifest written to: " + str(input_manifest_result.manifest_dir),
                    live_display,
                )
                if input_manifest_result.warning_count > 0:
                    important_info.add_text_line(
                        "Input manifest warnings: " + str(input_manifest_result.warning_count),
                        live_display,
                    )

                #live_display.stop()
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


                    


                    ### DOSE GRADIENT


                    # Scale the dose values before computing gradients
                    scaled_dose_data = dose_ref_dict["Dose pixel arr"] * dose_ref_dict["Dose grid scaling"]

                    gradient_vector_lattice, gradient_norm_lattice, normalized_gradient_vector_lattice = dose_lattice_helper_funcs.calculate_gradient_lattices(scaled_dose_data, dose_ref_dict["Pixel spacing"], dose_ref_dict["Grid frame offset vector"])
                    
                    
                    phys_space_dose_map_and_gradient_map_3d_arr = dose_lattice_helper_funcs.map_gradient_to_physical_space(
                        phys_space_dose_map_3d_arr=phys_space_dose_map_3d_arr,
                        gradient_vector_lattice=gradient_vector_lattice,
                        gradient_norm_lattice=gradient_norm_lattice,
                        normalized_gradient_vector_lattice = normalized_gradient_vector_lattice
                    )
                    """
                    Returns:
                        phys_space_dose_map_and_gradient_map_3d_arr (numpy.ndarray): Updated slice-wise array with gradients and normalized gradients added.
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













                #live_display.stop()
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
                        print(f"MR ADC render: {patientUID}")
                        plotting_funcs.plot_geometries(mr_adc_point_cloud)
                        stopwatch.start()
                        patients_progress.start_task(processing_patients_dose_task)
                        completed_progress.start_task(processing_patients_dose_task_completed)

                    
                    
                    # plot dose point cloud thresholded cubic lattice (color only)
                    if show_3d_mr_adc_renderings_thresholded == True:
                        patients_progress.stop_task(processing_patients_dose_task)
                        completed_progress.stop_task(processing_patients_dose_task_completed)
                        stopwatch.stop()
                        print(f"MR ADC render (tresholded): {patientUID}")
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
                    simulated_bx_rad = simulated_biopsy_planning_radius_mm
                    plot_simulated_cores_immediately = False
                """


                #live_display.stop()
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









                non_bx_structure_preprocessing_config = pipeline_config.preprocessing.build_non_biopsy_structure_preprocessing_config(
                    all_ref_key=all_ref_key,
                    oar_ref=oar_ref,
                    dil_ref=dil_ref,
                    mr_adc_ref=mr_adc_ref,
                )

                if validate_non_biopsy_structure_preprocessing_equivalence_bool != True:
                    live_display = process_standard_non_biopsy_structure_families(
                        oar_ref=oar_ref,
                        rectum_ref_key=rectum_ref_key,
                        urethra_ref_key=urethra_ref_key,
                        dil_ref=dil_ref,
                        master_structure_reference_dict=master_structure_reference_dict,
                        master_structure_info_dict=master_structure_info_dict,
                        structs_referenced_dict=structs_referenced_dict,
                        config=non_bx_structure_preprocessing_config,
                        parallel_pool=parallel_pool,
                        layout_groups=layout_groups,
                        patients_progress=patients_progress,
                        structures_progress=structures_progress,
                        completed_progress=completed_progress,
                        indeterminate_progress_sub=indeterminate_progress_sub,
                        important_info=important_info,
                        live_display=live_display,
                        runtime_logger=runtime_logger,
                    )
                else:
                    ### PREPROCESSING OARs


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

                            live_display, modular_validation_snapshot, modular_live_state = prepare_non_biopsy_structure_legacy_validation(
                                patient_uid=patientUID,
                                pydicom_item=pydicom_item,
                                master_structure_reference_dict=master_structure_reference_dict,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                structs_referenced_dict=structs_referenced_dict,
                                config=non_bx_structure_preprocessing_config,
                                parallel_pool=parallel_pool,
                                layout_groups=layout_groups,
                                structures_progress=structures_progress,
                                processing_structures_task=processing_structures_task,
                                indeterminate_progress_sub=indeterminate_progress_sub,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )

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

                            # old
                            #interpolation_information.create_fill(first_zslice, interp_dist_caps)
                            #interpolation_information.create_fill(last_zslice, interp_dist_caps)

                            # new
                            interpolation_information.create_fill_new_v2(first_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)
                            interpolation_information.create_fill_new_v2(last_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)

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

                            ### COMPUTE MR STATISTICS

                            if mr_adc_ref in pydicom_item:
                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (determining containment)", total = None)
                                ###
                                adc_mr_phys_space_arr = mr_localizers.grab_mr_adc_2d_arr(pydicom_item,
                                    mr_adc_ref,
                                    filter_out_negatives = True)
                                # Prepare data
                                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure)
                                #interslice_interpolation_information = specific_relative_structure["Inter-slice interpolation information"]
                                zslices_list = interslice_interpolation_information.interpolated_pts_list
                                mr_adc_value_column_name_str = "MR ADC value"
                                containment_info_for_all_lattice_points_grand_pandas_dataframe = mr_localizers.test_points_of_given_2d_lattice_from_within_given_structure_and_return_dataframe_type_2III(adc_mr_phys_space_arr,
                                                                    zslices_list,
                                                                    structure_info,
                                                                    constant_z_slice_polygons_handler_option,
                                                                    remove_consecutive_duplicate_points_in_polygons,
                                                                    custom_cuda_kernel_type,
                                                                    associated_value_str = mr_adc_value_column_name_str)
                                if demonstrate_mr_adc_pcd_containment_correctness_bool == True:
                                    plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                        "Test pt X",
                                                        "Test pt Y",
                                                        "Test pt Z",
                                                        "Pt clr R",
                                                        "Pt clr G",
                                                        "Pt clr B",
                                                        additional_point_clouds=[interpolated_pcd_dict['Full with end caps']])
                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###
                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (computing statistics)", total = None)
                                ###
                                # Create a summary statistics dataframe of the column
                                mr_adc_value_summary_statistics_specific_structure = dataframe_builders.dataframe_mr_summary_statistics(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                                                                                                        mr_adc_value_column_name_str,
                                                                                                                                        filter_column="Pt contained bool",
                                                                                                                                        filter_value=True)

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###
                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Keeping track of prostate only MR ADC values", total = None)
                                ###
                                # Keep track and store onky the points that are contained within the prostate (stored at end of loop)
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_true = containment_info_for_all_lattice_points_grand_pandas_dataframe[containment_info_for_all_lattice_points_grand_pandas_dataframe["Pt contained bool"] == True]

                                # Store it in the master dict
                                master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"] = containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_true

                                del containment_info_for_all_lattice_points_grand_pandas_dataframe

                                # if the following dataframe already exists, then merge the above with it by appending rows
                                if master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is not None:
                                    mr_adc_value_summary_statistics_specific_structure_master = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"]
                                    mr_adc_value_summary_statistics_specific_structure_master = pandas.concat([mr_adc_value_summary_statistics_specific_structure_master,
                                                                                                        mr_adc_value_summary_statistics_specific_structure],
                                                                                                        ignore_index = True)
                                    # Store the dataframe
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure_master

                                # if the following dataframe does not exist, then store the above dataframe
                                elif master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is None:
                                    # Store the dataframe if it does not exist
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure
                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###
                            ###### END COMPUTE MR STATISTICS

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
                                patientUID,
                                voxel_size_for_structure_volume_calc_non_bx,
                                factor_for_voxel_size,
                                cupy_array_upper_limit_NxN_size_input,
                                layout_groups,
                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                structures_progress,
                                live_display,
                                generate_cuda_log_files_volume_calculation = generate_cuda_log_files_volume_calculation,
                                constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                include_edges_in_log_files = include_edges_in_log_files,
                                custom_cuda_kernel_type = custom_cuda_kernel_type,
                                demonstrate_volume_calculation_correctness_bool_1 = demonstrate_volume_calculation_correctness_bool_1,
                                plot_volume_calculation_containment_result_bool_1_old = plot_volume_calculation_containment_result_bool_1_old,
                                plot_binary_mask_bool = plot_binary_mask_bool,
                                other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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
                                                                                                                            patientUID,
                                                                                                                            voxel_size_for_structure_dimension_calc,
                                                                                                                            factor_for_voxel_size,
                                                                                                                            cupy_array_upper_limit_NxN_size_input,
                                                                                                                            layout_groups,
                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                            structures_progress,
                                                                                                                            live_display,
                                                                                                                            generate_cuda_log_files_structure_dimension_calculation = generate_cuda_log_files_structure_dimension_calculation,
                                                                                                                            constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                                                                                                            remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                                                                                                            include_edges_in_log_files = include_edges_in_log_files,
                                                                                                                            custom_cuda_kernel_type = custom_cuda_kernel_type,
                                                                                                                            demonstrate_structure_dimension_calculation_correctness_bool_1 = demonstrate_structure_dimension_calculation_correctness_bool_1,
                                                                                                                            demonstrate_structure_dimension_calculation_correctness_bool_1_old = demonstrate_structure_dimension_calculation_correctness_bool_1_old,
                                                                                                                            other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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


                            fully_interp_with_end_caps_structure_triangle_mesh, water_tight_bool = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist,
                                interp_intra_slice_dist,
                                threeDdata_array_fully_interpolated_with_end_caps,
                                radius_for_normals_estimation,
                                max_nn_for_normals_estimation
                                )

                            if water_tight_bool == False:
                                important_info.add_text_line(f"WARNING! Patient: {patientUID}, Structure: {structureID}, ({structs}) is not water tight! Surface area may be inaccurate!", live_display)

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
                            si_arclength = misc_tools.compute_arc_length_from_centroids(specific_structure["Structure centroid pts"])


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
                                                            "Structure index": [specific_structure_index],
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
                                                            "S/I arclength": [si_arclength]
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

                            live_display = finalize_non_biopsy_structure_legacy_validation(
                                master_structure_reference_dict=master_structure_reference_dict,
                                patient_uid=patientUID,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                all_ref_key=all_ref_key,
                                structure_id=structureID,
                                modular_validation_snapshot=modular_validation_snapshot,
                                modular_live_state=modular_live_state,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )

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

                            live_display, modular_validation_snapshot, modular_live_state = prepare_non_biopsy_structure_legacy_validation(
                                patient_uid=patientUID,
                                pydicom_item=pydicom_item,
                                master_structure_reference_dict=master_structure_reference_dict,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                structs_referenced_dict=structs_referenced_dict,
                                config=non_bx_structure_preprocessing_config,
                                parallel_pool=parallel_pool,
                                layout_groups=layout_groups,
                                structures_progress=structures_progress,
                                processing_structures_task=processing_structures_task,
                                indeterminate_progress_sub=indeterminate_progress_sub,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )

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

                            # old
                            #interpolation_information.create_fill(first_zslice, interp_dist_caps)
                            #interpolation_information.create_fill(last_zslice, interp_dist_caps)

                            # new
                            interpolation_information.create_fill_new_v2(first_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)
                            interpolation_information.create_fill_new_v2(last_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)

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



                            ### COMPUTE MR STATISTICS

                            if mr_adc_ref in pydicom_item:

                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (determining containment)", total = None)
                                ###

                                adc_mr_phys_space_arr = mr_localizers.grab_mr_adc_2d_arr(pydicom_item,
                                    mr_adc_ref,
                                    filter_out_negatives = True)

                                # Prepare data
                                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure)
                                #interslice_interpolation_information = specific_relative_structure["Inter-slice interpolation information"]
                                zslices_list = interslice_interpolation_information.interpolated_pts_list
                                mr_adc_value_column_name_str = "MR ADC value"
                                containment_info_for_all_lattice_points_grand_pandas_dataframe = mr_localizers.test_points_of_given_2d_lattice_from_within_given_structure_and_return_dataframe_type_2III(adc_mr_phys_space_arr,
                                                                    zslices_list,
                                                                    structure_info,
                                                                    constant_z_slice_polygons_handler_option,
                                                                    remove_consecutive_duplicate_points_in_polygons,
                                                                    custom_cuda_kernel_type,
                                                                    associated_value_str = mr_adc_value_column_name_str)


                                if demonstrate_mr_adc_pcd_containment_correctness_bool == True:
                                    plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                        "Test pt X",
                                                        "Test pt Y",
                                                        "Test pt Z",
                                                        "Pt clr R",
                                                        "Pt clr G",
                                                        "Pt clr B",
                                                        additional_point_clouds=[interpolated_pcd_dict['Full with end caps']])

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###
                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (computing statistics)", total = None)
                                ###

                                # Create a summary statistics dataframe of the column
                                mr_adc_value_summary_statistics_specific_structure = dataframe_builders.dataframe_mr_summary_statistics(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                                                                                                        mr_adc_value_column_name_str,
                                                                                                                                        filter_column="Pt contained bool",
                                                                                                                                        filter_value=True)

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Keeping track of prostate only MR ADC values", total = None)
                                ###
                                # Keep track of the points that are ONLY in the prostate (ie with all other structure points removed)
                                # Retrieve
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"]
                                # remove the points from the prostate true dataframe that are contained true in the rectum data frame
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = dataframe_builders.drop_rows_where_b_is_true(
                                                    containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only,
                                                    containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                    index_col= "Test pt index",
                                                    flag_col= "Pt contained bool",
                                                    keep_unmatched = True
                                                )

                                master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"] = containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only

                                del containment_info_for_all_lattice_points_grand_pandas_dataframe

                                # if the following dataframe already exists, then merge the above with it by appending rows
                                if master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is not None:

                                    mr_adc_value_summary_statistics_specific_structure_master = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"]
                                    mr_adc_value_summary_statistics_specific_structure_master = pandas.concat([mr_adc_value_summary_statistics_specific_structure_master,
                                                                                                        mr_adc_value_summary_statistics_specific_structure],
                                                                                                        ignore_index = True)
                                    # Store the dataframe
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure_master

                                # if the following dataframe does not exist, then store the above dataframe
                                elif master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is None:
                                    # Store the dataframe if it does not exist
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                            ###### END COMPUTE MR STATISTICS





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
                                patientUID,
                                voxel_size_for_structure_volume_calc_non_bx,
                                factor_for_voxel_size,
                                cupy_array_upper_limit_NxN_size_input,
                                layout_groups,
                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                structures_progress,
                                live_display,
                                generate_cuda_log_files_volume_calculation = generate_cuda_log_files_volume_calculation,
                                constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                include_edges_in_log_files = include_edges_in_log_files,
                                custom_cuda_kernel_type = custom_cuda_kernel_type,
                                demonstrate_volume_calculation_correctness_bool_1 = demonstrate_volume_calculation_correctness_bool_1,
                                plot_volume_calculation_containment_result_bool_1_old = plot_volume_calculation_containment_result_bool_1_old,
                                plot_binary_mask_bool = plot_binary_mask_bool,
                                other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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
                                                                                                                            patientUID,
                                                                                                                            voxel_size_for_structure_dimension_calc,
                                                                                                                            factor_for_voxel_size,
                                                                                                                            cupy_array_upper_limit_NxN_size_input,
                                                                                                                            layout_groups,
                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                            structures_progress,
                                                                                                                            live_display,
                                                                                                                            generate_cuda_log_files_structure_dimension_calculation = generate_cuda_log_files_structure_dimension_calculation,
                                                                                                                            constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                                                                                                            remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                                                                                                            include_edges_in_log_files = include_edges_in_log_files,
                                                                                                                            custom_cuda_kernel_type = custom_cuda_kernel_type,
                                                                                                                            demonstrate_structure_dimension_calculation_correctness_bool_1 = demonstrate_structure_dimension_calculation_correctness_bool_1,
                                                                                                                            demonstrate_structure_dimension_calculation_correctness_bool_1_old = demonstrate_structure_dimension_calculation_correctness_bool_1_old,
                                                                                                                            other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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

                            fully_interp_with_end_caps_structure_triangle_mesh, water_tight_bool = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist,
                                interp_intra_slice_dist,
                                threeDdata_array_fully_interpolated_with_end_caps,
                                radius_for_normals_estimation,
                                max_nn_for_normals_estimation
                                )

                            if water_tight_bool == False:
                                important_info.add_text_line(f"WARNING! Patient: {patientUID}, Structure: {structureID}, ({structs}) is not water tight! Surface area may be inaccurate!", live_display)

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
                            si_arclength = misc_tools.compute_arc_length_from_centroids(specific_structure["Structure centroid pts"])

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
                                                            "Structure index": [specific_structure_index],
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
                                                            "S/I arclength": [si_arclength]
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

                            live_display = finalize_non_biopsy_structure_legacy_validation(
                                master_structure_reference_dict=master_structure_reference_dict,
                                patient_uid=patientUID,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                all_ref_key=all_ref_key,
                                structure_id=structureID,
                                modular_validation_snapshot=modular_validation_snapshot,
                                modular_live_state=modular_live_state,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )


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

                            live_display, modular_validation_snapshot, modular_live_state = prepare_non_biopsy_structure_legacy_validation(
                                patient_uid=patientUID,
                                pydicom_item=pydicom_item,
                                master_structure_reference_dict=master_structure_reference_dict,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                structs_referenced_dict=structs_referenced_dict,
                                config=non_bx_structure_preprocessing_config,
                                parallel_pool=parallel_pool,
                                layout_groups=layout_groups,
                                structures_progress=structures_progress,
                                processing_structures_task=processing_structures_task,
                                indeterminate_progress_sub=indeterminate_progress_sub,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )

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

                            # old
                            #interpolation_information.create_fill(first_zslice, interp_dist_caps)
                            #interpolation_information.create_fill(last_zslice, interp_dist_caps)

                            # new
                            interpolation_information.create_fill_new_v2(first_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)
                            interpolation_information.create_fill_new_v2(last_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)

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





                            ### COMPUTE MR STATISTICS

                            if mr_adc_ref in pydicom_item:

                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (determining containment)", total = None)
                                ###

                                adc_mr_phys_space_arr = mr_localizers.grab_mr_adc_2d_arr(pydicom_item,
                                    mr_adc_ref,
                                    filter_out_negatives = True)

                                # Prepare data
                                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure)
                                #interslice_interpolation_information = specific_relative_structure["Inter-slice interpolation information"]
                                zslices_list = interslice_interpolation_information.interpolated_pts_list
                                mr_adc_value_column_name_str = "MR ADC value"
                                containment_info_for_all_lattice_points_grand_pandas_dataframe = mr_localizers.test_points_of_given_2d_lattice_from_within_given_structure_and_return_dataframe_type_2III(adc_mr_phys_space_arr,
                                                                    zslices_list,
                                                                    structure_info,
                                                                    constant_z_slice_polygons_handler_option,
                                                                    remove_consecutive_duplicate_points_in_polygons,
                                                                    custom_cuda_kernel_type,
                                                                    associated_value_str = mr_adc_value_column_name_str)


                                if demonstrate_mr_adc_pcd_containment_correctness_bool == True:
                                    plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                        "Test pt X",
                                                        "Test pt Y",
                                                        "Test pt Z",
                                                        "Pt clr R",
                                                        "Pt clr G",
                                                        "Pt clr B",
                                                        additional_point_clouds=[interpolated_pcd_dict['Full with end caps']])

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###
                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (computing statistics)", total = None)
                                ###

                                # Create a summary statistics dataframe of the column
                                mr_adc_value_summary_statistics_specific_structure = dataframe_builders.dataframe_mr_summary_statistics(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                                                                                                        mr_adc_value_column_name_str,
                                                                                                                                        filter_column="Pt contained bool",
                                                                                                                                        filter_value=True)


                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Keeping track of prostate only MR ADC values", total = None)
                                ###
                                # Keep track of the points that are ONLY in the prostate (ie with all other structure points removed)
                                # Retrieve
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"]
                                # remove the points from the prostate true dataframe that are contained true in the rectum data frame
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = dataframe_builders.drop_rows_where_b_is_true(
                                                    containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only,
                                                    containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                    index_col= "Test pt index",
                                                    flag_col= "Pt contained bool",
                                                    keep_unmatched = True
                                                )
                                master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"] = containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only
                                del containment_info_for_all_lattice_points_grand_pandas_dataframe

                                # if the following dataframe already exists, then merge the above with it by appending rows
                                if master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is not None:

                                    mr_adc_value_summary_statistics_specific_structure_master = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"]
                                    mr_adc_value_summary_statistics_specific_structure_master = pandas.concat([mr_adc_value_summary_statistics_specific_structure_master,
                                                                                                        mr_adc_value_summary_statistics_specific_structure],
                                                                                                        ignore_index = True)
                                    # Store the dataframe
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure_master

                                # if the following dataframe does not exist, then store the above dataframe
                                elif master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is None:
                                    # Store the dataframe if it does not exist
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                            ###### END COMPUTE MR STATISTICS





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
                                patientUID,
                                voxel_size_for_structure_volume_calc_non_bx,
                                factor_for_voxel_size,
                                cupy_array_upper_limit_NxN_size_input,
                                layout_groups,
                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                structures_progress,
                                live_display,
                                generate_cuda_log_files_volume_calculation = generate_cuda_log_files_volume_calculation,
                                constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                include_edges_in_log_files = include_edges_in_log_files,
                                custom_cuda_kernel_type = custom_cuda_kernel_type,
                                demonstrate_volume_calculation_correctness_bool_1 = demonstrate_volume_calculation_correctness_bool_1,
                                plot_volume_calculation_containment_result_bool_1_old = plot_volume_calculation_containment_result_bool_1_old,
                                plot_binary_mask_bool = plot_binary_mask_bool,
                                other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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
                                                                                                                            patientUID,
                                                                                                                            voxel_size_for_structure_dimension_calc,
                                                                                                                            factor_for_voxel_size,
                                                                                                                            cupy_array_upper_limit_NxN_size_input,
                                                                                                                            layout_groups,
                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                            structures_progress,
                                                                                                                            live_display,
                                                                                                                            generate_cuda_log_files_structure_dimension_calculation = generate_cuda_log_files_structure_dimension_calculation,
                                                                                                                            constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                                                                                                            remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                                                                                                            include_edges_in_log_files = include_edges_in_log_files,
                                                                                                                            custom_cuda_kernel_type = custom_cuda_kernel_type,
                                                                                                                            demonstrate_structure_dimension_calculation_correctness_bool_1 = demonstrate_structure_dimension_calculation_correctness_bool_1,
                                                                                                                            demonstrate_structure_dimension_calculation_correctness_bool_1_old = demonstrate_structure_dimension_calculation_correctness_bool_1_old,
                                                                                                                            other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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

                            fully_interp_with_end_caps_structure_triangle_mesh, water_tight_bool = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist,
                                interp_intra_slice_dist,
                                threeDdata_array_fully_interpolated_with_end_caps,
                                radius_for_normals_estimation,
                                max_nn_for_normals_estimation
                                )

                            if water_tight_bool == False:
                                important_info.add_text_line(f"WARNING! Patient: {patientUID}, Structure: {structureID}, ({structs}) is not water tight! Surface area may be inaccurate!", live_display)

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
                            si_arclength = misc_tools.compute_arc_length_from_centroids(specific_structure["Structure centroid pts"])

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
                                                            "Structure index": [specific_structure_index],
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
                                                            "S/I arclength": [si_arclength]
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

                            live_display = finalize_non_biopsy_structure_legacy_validation(
                                master_structure_reference_dict=master_structure_reference_dict,
                                patient_uid=patientUID,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                all_ref_key=all_ref_key,
                                structure_id=structureID,
                                modular_validation_snapshot=modular_validation_snapshot,
                                modular_live_state=modular_live_state,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )



                            structures_progress.update(processing_structures_task, advance=1)

                        structures_progress.remove_task(processing_structures_task)
                        patients_progress.update(processing_patients_task, advance=1)
                        completed_progress.update(processing_patients_task_completed, advance=1)
                    patients_progress.update(processing_patients_task, visible=False)
                    completed_progress.update(processing_patients_task_completed,  visible=True)





















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

                            live_display, modular_validation_snapshot, modular_live_state = prepare_non_biopsy_structure_legacy_validation(
                                patient_uid=patientUID,
                                pydicom_item=pydicom_item,
                                master_structure_reference_dict=master_structure_reference_dict,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                structs_referenced_dict=structs_referenced_dict,
                                config=non_bx_structure_preprocessing_config,
                                parallel_pool=parallel_pool,
                                layout_groups=layout_groups,
                                structures_progress=structures_progress,
                                processing_structures_task=processing_structures_task,
                                indeterminate_progress_sub=indeterminate_progress_sub,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                                sp_patient_selected_structure_info_dataframe=(
                                    sp_patient_selected_structure_info_dataframe
                                ),
                            )

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

                            # old
                            #interpolation_information.create_fill(first_zslice, interp_dist_caps)
                            #interpolation_information.create_fill(last_zslice, interp_dist_caps)

                            # new
                            interpolation_information.create_fill_new_v2(first_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)
                            interpolation_information.create_fill_new_v2(last_zslice, interp_dist_caps, kernel_type=custom_cuda_kernel_type)

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





                            ### COMPUTE MR STATISTICS

                            if mr_adc_ref in pydicom_item:

                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (determining containment)", total = None)
                                ###

                                adc_mr_phys_space_arr = mr_localizers.grab_mr_adc_2d_arr(pydicom_item,
                                    mr_adc_ref,
                                    filter_out_negatives = True)

                                # Prepare data
                                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure)
                                #interslice_interpolation_information = specific_relative_structure["Inter-slice interpolation information"]
                                zslices_list = interslice_interpolation_information.interpolated_pts_list
                                mr_adc_value_column_name_str = "MR ADC value"
                                containment_info_for_all_lattice_points_grand_pandas_dataframe = mr_localizers.test_points_of_given_2d_lattice_from_within_given_structure_and_return_dataframe_type_2III(adc_mr_phys_space_arr,
                                                                    zslices_list,
                                                                    structure_info,
                                                                    constant_z_slice_polygons_handler_option,
                                                                    remove_consecutive_duplicate_points_in_polygons,
                                                                    custom_cuda_kernel_type,
                                                                    associated_value_str = mr_adc_value_column_name_str)


                                if demonstrate_mr_adc_pcd_containment_correctness_bool == True:
                                    plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                        "Test pt X",
                                                        "Test pt Y",
                                                        "Test pt Z",
                                                        "Pt clr R",
                                                        "Pt clr G",
                                                        "Pt clr B",
                                                        additional_point_clouds=[interpolated_pcd_dict['Full with end caps']])

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###
                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calculating MR statistics (computing statistics)", total = None)
                                ###

                                # Create a summary statistics dataframe of the column
                                mr_adc_value_summary_statistics_specific_structure = dataframe_builders.dataframe_mr_summary_statistics(containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                                                                                                        mr_adc_value_column_name_str,
                                                                                                                                        filter_column="Pt contained bool",
                                                                                                                                        filter_value=True)

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                                ###
                                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Keeping track of prostate only MR ADC values", total = None)
                                ###
                                # Keep track of the points that are ONLY in the prostate (ie with all other structure points removed)
                                # Retrieve
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"]
                                # remove the points from the prostate true dataframe that are contained true in the rectum data frame
                                containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only = dataframe_builders.drop_rows_where_b_is_true(
                                                    containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only,
                                                    containment_info_for_all_lattice_points_grand_pandas_dataframe,
                                                    index_col= "Test pt index",
                                                    flag_col= "Pt contained bool",
                                                    keep_unmatched = True
                                                )
                                master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Prostate only points MR ADC dataframe (temporary for pre-processing)"] = containment_info_for_all_lattice_points_grand_pandas_dataframe_prostate_only


                                del containment_info_for_all_lattice_points_grand_pandas_dataframe

                                # if the following dataframe already exists, then merge the above with it by appending rows
                                if master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is not None:

                                    mr_adc_value_summary_statistics_specific_structure_master = master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"]
                                    mr_adc_value_summary_statistics_specific_structure_master = pandas.concat([mr_adc_value_summary_statistics_specific_structure_master,
                                                                                                        mr_adc_value_summary_statistics_specific_structure],
                                                                                                        ignore_index = True)
                                    # Store the dataframe
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure_master

                                # if the following dataframe does not exist, then store the above dataframe
                                elif master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] is None:
                                    # Store the dataframe if it does not exist
                                    master_structure_reference_dict[patientUID][all_ref_key]["Multi-structure pre-processing output dataframes dict"]["MR - ADC - summary statistics by structure dataframe"] = mr_adc_value_summary_statistics_specific_structure

                                ###
                                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                                ###

                            ###### END COMPUTE MR STATISTICS













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
                                patientUID,
                                voxel_size_for_structure_volume_calc_non_bx,
                                factor_for_voxel_size,
                                cupy_array_upper_limit_NxN_size_input,
                                layout_groups,
                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                structures_progress,
                                live_display,
                                generate_cuda_log_files_volume_calculation = generate_cuda_log_files_volume_calculation,
                                constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                include_edges_in_log_files = include_edges_in_log_files,
                                custom_cuda_kernel_type = custom_cuda_kernel_type,
                                demonstrate_volume_calculation_correctness_bool_1 = demonstrate_volume_calculation_correctness_bool_1,
                                plot_volume_calculation_containment_result_bool_1_old = plot_volume_calculation_containment_result_bool_1_old,
                                plot_binary_mask_bool = plot_binary_mask_bool,
                                other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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
                                                                                                                        patientUID,
                                                                                                                        voxel_size_for_structure_dimension_calc,
                                                                                                                        factor_for_voxel_size,
                                                                                                                        cupy_array_upper_limit_NxN_size_input,
                                                                                                                        layout_groups,
                                                                                                                        nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                        structures_progress,
                                                                                                                        live_display,
                                                                                                                        generate_cuda_log_files_structure_dimension_calculation = generate_cuda_log_files_structure_dimension_calculation,
                                                                                                                        constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                                                                                                        remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                                                                                                        include_edges_in_log_files = include_edges_in_log_files,
                                                                                                                        custom_cuda_kernel_type = custom_cuda_kernel_type,
                                                                                                                        demonstrate_structure_dimension_calculation_correctness_bool_1 = demonstrate_structure_dimension_calculation_correctness_bool_1,
                                                                                                                        demonstrate_structure_dimension_calculation_correctness_bool_1_old = demonstrate_structure_dimension_calculation_correctness_bool_1_old,
                                                                                                                        other_pcds_to_plot_list = [interpolated_pcd_dict['Full with end caps']]
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

                            fully_interp_with_end_caps_structure_triangle_mesh, water_tight_bool = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist,
                                interp_intra_slice_dist,
                                threeDdata_array_fully_interpolated_with_end_caps,
                                radius_for_normals_estimation,
                                max_nn_for_normals_estimation
                                )

                            if water_tight_bool == False:
                                important_info.add_text_line(f"WARNING! Patient: {patientUID}, Structure: {structureID}, ({structs}) is not water tight! Surface area may be inaccurate!", live_display)

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
                            si_arclength = misc_tools.compute_arc_length_from_centroids(specific_structure["Structure centroid pts"])

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
                                                            "Structure index": [specific_structure_index],
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
                                                            "S/I arclength": [si_arclength],
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

                            live_display = finalize_non_biopsy_structure_legacy_validation(
                                master_structure_reference_dict=master_structure_reference_dict,
                                patient_uid=patientUID,
                                struct_ref_type=structs,
                                specific_structure_index=specific_structure_index,
                                all_ref_key=all_ref_key,
                                structure_id=structureID,
                                modular_validation_snapshot=modular_validation_snapshot,
                                modular_live_state=modular_live_state,
                                important_info=important_info,
                                live_display=live_display,
                                runtime_logger=runtime_logger,
                            )



                            structures_progress.update(processing_structures_task, advance=1)

                        structures_progress.remove_task(processing_structures_task)
                        patients_progress.update(processing_patients_task, advance=1)
                        completed_progress.update(processing_patients_task_completed, advance=1)
                    patients_progress.update(processing_patients_task, visible=False)
                    completed_progress.update(processing_patients_task_completed,  visible=True)


                    ### END DIL STRUCTURE PROCESSING

                



















                ### CALCULATE PROSTATE ONLY MR ADC VALUES WITH DILS, RECTUM URETHRA POINTS REMOVED

                live_display = prostate_only_mr_adc_processer(
                    master_structure_reference_dict,
                    master_structure_info_dict,
                    all_ref_key,
                    dil_ref,
                    mr_adc_ref,
                    demonstrate_mr_adc_pcd_containment_correctness_prostate_only_all_other_structures_removed_bool,
                    patients_progress,
                    completed_progress,
                    indeterminate_progress_sub,
                    live_display,
                )

                ### END CALCULATING PROSTATE ONLY MR ADC VALUES WITH DILS, RECTUM URETHRA POINTS REMOVED





                






                #### New spot for real biopsy processor
                live_display = real_biopsy_processer(master_structure_reference_dict,
                            master_structure_info_dict,
                            structs_referenced_dict,
                            bx_ref,
                            dil_ref,
                            all_ref_key,
                            parallel_pool,
                            interp_inter_slice_dist,
                            interp_intra_slice_dist,
                            interp_dist_caps,
                            biopsy_radius,
                            display_pca_fit_variation_for_biopsies_bool,
                            voxel_size_for_structure_volume_calc_non_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            generate_cuda_log_files_volume_calculation,
                            constant_z_slice_polygons_handler_option,
                            remove_consecutive_duplicate_points_in_polygons,
                            include_edges_in_log_files,
                            custom_cuda_kernel_type,
                            demonstrate_volume_calculation_correctness_bool_1,
                            plot_volume_calculation_containment_result_bool_1_old,
                            plot_binary_mask_bool,
                            patients_progress,
                            structures_progress,
                            completed_progress,
                            indeterminate_progress_sub,
                            live_display
                            )

                live_display = simulated_biopsy_preparer(master_structure_reference_dict,
                            bx_ref,
                            dil_ref,
                            all_ref_key,
                            simulated_biopsy_length_method,
                            biopsy_needle_compartment_length,
                            live_display,
                            master_structure_info_dict = master_structure_info_dict
                            )

                ###################    SET SOME PRELIMS FOR THE SIMULATED BIOPSIES

                # initialize the basics for drawing the simulated biopsies
                centroid_line_vec_sim_list = [0,0,1]
                centroid_first_pos_sim_list = [0,0,0]
                num_centroids_for_sim_bxs = 10
                simulated_bx_rad = simulated_biopsy_planning_radius_mm
                plot_simulated_cores_immediately = False
                # note that the length of the simulated biopsy is determined on a per biopsy basis in the below code!

                live_display = simulated_biopsy_planner_processer(master_structure_reference_dict,
                            master_structure_info_dict,
                            bx_ref,
                            bx_sample_pts_lattice_spacing,
                            parallel_pool,
                            patients_progress,
                            structures_progress,
                            completed_progress,
                            live_display,
                            centroid_line_vec_sim_list,
                            centroid_first_pos_sim_list,
                            num_centroids_for_sim_bxs,
                            simulated_bx_rad,
                            plot_simulated_cores_immediately,
                            )

                uncertainties_file, uncertainties_file_filled, read_uncertainties_dataframe, live_display = prepare_and_attach_uncertainty_data(
                            master_structure_reference_dict,
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
                            uncertainty_data,
                            )

                max_simulations, max_generated_transform_samples = configure_transform_generation_counts(
                            master_structure_info_dict,
                            num_MC_containment_simulations_input,
                            num_MC_dose_simulations_input,
                            num_MC_MR_simulations_input,
                            )

                if max_generated_transform_samples > 0:
                    indeterminate_task_generating_transforms = indeterminate_progress_main.add_task("[red]Generating transforms", total=None)
                    indeterminate_task_generating_transforms_completed = completed_progress.add_task("[green]Generating transforms", visible = False, total = 1)

                    transform_sampling_rng = build_transform_generation_rng(master_structure_info_dict)
                    MC_prepper_funcs.generate_transformations(master_structure_reference_dict,
                                                    simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                                    bx_ref,
                                                    biopsy_needle_compartment_length,
                                                    max_generated_transform_samples,
                                                    structs_referenced_list,
                                                    rng=transform_sampling_rng)

                    indeterminate_progress_main.update(indeterminate_task_generating_transforms, visible = False, refresh = True)
                    completed_progress.update(indeterminate_task_generating_transforms_completed, advance = 1, visible = True, refresh = True)
                    live_display.refresh()


                ########## PERFORM BIOPSY DIL OPTIMIZATION
                # modularized!
                runtime_logger.checkpoint(
                    "optimizer.preflight",
                    "Keeping live display active during optimizer stages.",
                )
                runtime_logger.memory_snapshot(
                    "optimizer_v1.pre",
                    "Captured memory snapshot before optimizer-v1.",
                )
                runtime_logger.phase_start("optimizer_v1", "Starting optimizer-v1.")
                apply_optimizer_v1_random_seed(master_structure_info_dict)
                live_display = biopsy_optimizer_module_v1(master_structure_reference_dict,
                              master_structure_info_dict,
                              structs_referenced_dict,
                              bx_ref,
                              dil_ref,
                              oar_ref,
                              all_ref_key,
                              voxel_size_for_dil_optimizer_grid,
                              optimal_normal_dist_option,
                              bias_LR_multiplier,
                              bias_AP_multiplier,
                              bias_SI_multiplier,
                              num_normal_dist_points_for_biopsy_optimizer,
                              normal_dist_sigma_factor_biopsy_optimizer,
                              plot_each_normal_dist_containment_result_bool,
                              plot_optimization_point_lattice_bool,
                              show_optimization_point_bool,
                              cupy_array_upper_limit_NxN_size_input,
                              numpy_array_upper_limit_NxN_size_input,
                              nearest_zslice_vals_and_indices_cupy_generic_max_size,
                              nearest_zslice_vals_and_indices_numpy_generic_max_size,
                              constant_z_slice_polygons_handler_option,
                              remove_consecutive_duplicate_points_in_polygons,
                              include_edges_in_log_files,
                              custom_cuda_kernel_type,
                              demonstrate_dil_optimization_points_inside_correctness_bool_1,
                              demonstrate_dil_optimization_points_inside_correctness_bool_2,
                              demonstrate_dil_optimization_points_inside_correctness_num_3,
                              generate_cuda_log_files_biopsy_optimizer,
                              display_optimization_contour_plots_bool,
                              layout_groups,
                              patients_progress,
                              structures_progress,
                              indeterminate_progress_sub,
                              important_info,
                              completed_progress,
                              live_display,
                              )
                runtime_logger.phase_end("optimizer_v1", "Completed optimizer-v1.")
                runtime_logger.memory_snapshot(
                    "optimizer_v1.post",
                    "Captured memory snapshot after optimizer-v1.",
                )

                runtime_logger.phase_start("optimizer_v2", "Starting optimizer-v2.")
                live_display = run_target_dil_optimizer_v2_for_live_simulated_family(
                              master_structure_reference_dict,
                              master_structure_info_dict,
                              structs_referenced_dict,
                              bx_ref,
                              dil_ref,
                              all_ref_key,
                              target_dil_v2_sim_key,
                              optimizer_v2_search_config,
                              parallel_pool,
                              constant_z_slice_polygons_handler_option,
                              remove_consecutive_duplicate_points_in_polygons,
                              include_edges_in_log_files,
                              custom_cuda_kernel_type,
                              patients_progress,
                              structures_progress,
                              completed_progress,
                              live_display,
                              max_candidates_per_chunk=optimizer_v2_max_candidates_per_chunk,
                              max_test_structures_per_call=optimizer_v2_max_test_structures_per_call,
                              fallback_max_test_structures_per_call=optimizer_v2_fallback_max_test_structures_per_call,
                              auto_calibrate_max_test_structures_per_call=optimizer_v2_auto_calibrate_max_test_structures_per_call,
                              verify_calibrated_max_test_structures_per_call=(
                                  optimizer_v2_verify_calibrated_max_test_structures_per_call
                              ),
                              validate_nearest_z_helper_against_ver5=optimizer_v2_validate_nearest_z_helper_against_ver5_bool,
                              downstream_comparable_trial_count=(
                                  int(num_MC_containment_simulations_input)
                                  if num_MC_containment_simulations_input > 0
                                  else None
                              ),
                              benchmark_isolated_winner_validation_bool=optimizer_v2_benchmark_isolated_winner_validation_bool,
                              render_stage_boundary_candidate_clouds_bool=optimizer_v2_render_stage_boundary_candidate_clouds_bool,
                              render_stage_names_to_render=optimizer_v2_render_stage_names,
                              render_backend=optimizer_v2_render_backend,
                              render_layer_style_by_name=optimizer_v2_render_layer_style_by_name,
                              render_plotly_export_bool=optimizer_v2_render_plotly_export_bool,
                              render_plotly_export_formats=optimizer_v2_render_plotly_export_formats,
                              render_plotly_export_width=optimizer_v2_render_plotly_export_width,
                              render_plotly_export_height=optimizer_v2_render_plotly_export_height,
                              render_plotly_export_scale=optimizer_v2_render_plotly_export_scale,
                              render_plotly_export_camera_eye=optimizer_v2_render_plotly_export_camera_eye,
                              render_plotly_export_camera_center=optimizer_v2_render_plotly_export_camera_center,
                              render_plotly_export_camera_up=optimizer_v2_render_plotly_export_camera_up,
                              render_dialog_timeout_seconds=optimizer_v2_render_dialog_timeout_seconds,
                              render_dialog_timeout_extend_seconds=optimizer_v2_render_dialog_timeout_extend_seconds,
                              render_winner_containment_debug_bool=optimizer_v2_render_winner_containment_debug_bool,
                              render_winner_containment_backend=optimizer_v2_render_winner_containment_backend,
                              render_include_target_points_bool=optimizer_v2_render_include_target_points_bool,
                              render_include_target_surface_bool=optimizer_v2_render_include_target_surface_bool,
                              render_patient_whitelist=optimizer_v2_render_patient_whitelist,
                              render_roi_whitelist=optimizer_v2_render_roi_whitelist,
                              oar_ref=oar_ref,
                              rectum_ref=rectum_ref_key,
                              urethra_ref=urethra_ref_key,
                              )
                runtime_logger.phase_end("optimizer_v2", "Completed optimizer-v2.")
                runtime_logger.memory_snapshot(
                    "optimizer_v2.post",
                    "Captured memory snapshot after optimizer-v2.",
                )

                
                #####DONE##### PERFORM BIOPSY DIL OPTIMIZATION







                ###################    FINALIZE SIMULATED BIOPSIES FROM PREPARED/PLANNED STATE

                live_display = simulated_biopsy_processer(master_structure_reference_dict,
                            master_structure_info_dict,
                            structs_referenced_dict,
                            bx_ref,
                            parallel_pool,
                            interp_inter_slice_dist,
                            interp_intra_slice_dist,
                            interp_dist_caps,
                            biopsy_radius,
                            voxel_size_for_structure_volume_calc_non_bx,
                            factor_for_voxel_size,
                            cupy_array_upper_limit_NxN_size_input,
                            layout_groups,
                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                            generate_cuda_log_files_volume_calculation,
                            constant_z_slice_polygons_handler_option,
                            remove_consecutive_duplicate_points_in_polygons,
                            include_edges_in_log_files,
                            custom_cuda_kernel_type,
                            demonstrate_volume_calculation_correctness_bool_1,
                            plot_volume_calculation_containment_result_bool_1_old,
                            plot_binary_mask_bool,
                            patients_progress,
                            structures_progress,
                            completed_progress,
                            indeterminate_progress_sub,
                            live_display,
                            )

                simulated_biopsy_centroid_variation_validation_dataframe, simulated_biopsy_centroid_variation_validation_summary_dict = validate_simulated_biopsy_planned_vs_realized_centroid_variation(
                    master_structure_reference_dict,
                    bx_ref,
                    all_ref_key,
                )
                master_cohort_patient_data_and_dataframes["Dataframes"][
                    "Cohort: Simulated biopsy planned vs realized centroid variation validation"
                ] = simulated_biopsy_centroid_variation_validation_dataframe

                num_simulated_biopsies_validated = simulated_biopsy_centroid_variation_validation_summary_dict["Num simulated biopsies"]
                num_missing_validation_values = (
                    simulated_biopsy_centroid_variation_validation_summary_dict["Num missing planned mean centroid variation"]
                    + simulated_biopsy_centroid_variation_validation_summary_dict["Num missing realized mean centroid variation"]
                    + simulated_biopsy_centroid_variation_validation_summary_dict["Num missing planned maximum projected distance"]
                    + simulated_biopsy_centroid_variation_validation_summary_dict["Num missing realized maximum projected distance"]
                )

                if num_simulated_biopsies_validated > 0:
                    important_info.add_text_line(
                        "Simulated biopsy centroid-variation validation: compared {} simulated biopsies | mean abs delta (mean variation) = {} | max abs delta (mean variation) = {} | mean abs delta (max projected distance) = {} | max abs delta (max projected distance) = {}.".format(
                            num_simulated_biopsies_validated,
                            simulated_biopsy_centroid_variation_validation_summary_dict["Mean mean-centroid-variation absolute delta"],
                            simulated_biopsy_centroid_variation_validation_summary_dict["Max mean-centroid-variation absolute delta"],
                            simulated_biopsy_centroid_variation_validation_summary_dict["Mean max-projected-distance absolute delta"],
                            simulated_biopsy_centroid_variation_validation_summary_dict["Max max-projected-distance absolute delta"],
                        ),
                        live_display,
                    )

                if num_missing_validation_values > 0:
                    important_info.add_text_line(
                        "Notice! Simulated biopsy centroid-variation validation found missing planned or realized comparison values for {} fields across simulated biopsies.".format(
                            num_missing_validation_values,
                        ),
                        live_display,
                    )

                
                ################## ALL BIOPSIES 

                ################# BIOPSY TARGETTING

                live_display = realized_biopsy_targeting_processer(
                    master_structure_reference_dict,
                    master_structure_info_dict,
                    all_ref_key,
                    bx_ref,
                    oar_ref,
                    dil_ref,
                    patients_progress,
                    structures_progress,
                    completed_progress,
                    live_display,
                )


                master_structure_info_dict["Global"]['Preprocessing info']["Preprocessing performed"] = True
                ## END PREPROCESSING             

                
                #live_display.stop()

                ## The preprocessed bundle exporter prunes the in-memory structure tree back to the post-preprocessing boundary before pickling.

                # Now can export the preprocessing-bounded master structure dict to file.
                if pipeline_config.artifacts.export_pickled_preprocessed_data == True:
                    export_preprocessed_data_task_indeterminate = indeterminate_progress_main.add_task("[red]Exporting preprocessed data...", total=None)
                    export_preprocessed_data_task_indeterminate_completed = completed_progress.add_task("[green]Exporting preprocessed data", visible = False, total=master_structure_info_dict["Global"]["Num cases"])
                    
                    date_time_now = datetime.now()
                    date_time_now_file_name_format = date_time_now.strftime(" Date-%b-%d-%Y Time-%H,%M,%S")
                    global_num_structures = master_structure_info_dict["Global"]["Num structures"]
                    specific_preprocessed_data_dir_name = str(master_structure_info_dict["Global"]["Num cases"])+' patients - '+str(global_num_structures)+' structures - '+date_time_now_file_name_format
                    specific_preprocessed_data_dir = preprocessed_data_dir.joinpath(specific_preprocessed_data_dir_name)
                    specific_preprocessed_data_dir.mkdir(parents=False, exist_ok=False)
                    preprocessed_info_file_name = str(master_structure_info_dict["Global"]["Num cases"])+' patients - '+str(global_num_structures)+' structures.csv'
                    runtime_logger.phase_start(
                        "pickle_export.preprocessed",
                        "Exporting preprocessed pickle bundle.",
                        details={
                            "specific_preprocessed_data_dir": specific_preprocessed_data_dir,
                            "num_cases": master_structure_info_dict["Global"]["Num cases"],
                            "num_structures": global_num_structures,
                        },
                    )
                    export_preprocessed_pickle_bundle(
                        master_structure_reference_dict,
                        master_structure_info_dict,
                        specific_preprocessed_data_dir,
                        pipeline_config.artifacts.preprocessed_reference_dict_filename,
                        pipeline_config.artifacts.preprocessed_info_dict_filename,
                        preprocessed_info_file_name,
                        structs_referenced_list,
                        pipeline_config.preprocessing.build_frozen_preprocessed_bundle_config(),
                        bx_ref,
                        oar_ref,
                        dil_ref,
                        rectum_ref_key,
                        urethra_ref_key,
                        dose_ref,
                        mr_adc_ref,
                    )
                    runtime_logger.phase_end(
                        "pickle_export.preprocessed",
                        "Exported preprocessed pickle bundle.",
                        details={"specific_preprocessed_data_dir": specific_preprocessed_data_dir},
                    )


                    indeterminate_progress_main.update(export_preprocessed_data_task_indeterminate, visible = False, refresh = True)
                    completed_progress.update(export_preprocessed_data_task_indeterminate_completed, visible = True, refresh = True, advance=master_structure_info_dict["Global"]["Num cases"])
                    live_display.refresh()
                else:
                    export_preprocessed_data_task_indeterminate_skipped_completed = completed_progress.add_task("[green]Exporting preprocessed data [SKIPPED]", visible = False, total=None)
                    completed_progress.stop_task(export_preprocessed_data_task_indeterminate_skipped_completed)
                    completed_progress.update(export_preprocessed_data_task_indeterminate_skipped_completed, visible = True, refresh = True)
                    live_display.refresh()

                
                

            elif pipeline_config.artifacts.skip_preprocessing == True:
                live_display.stop()
                live_display.console.print("[bold red]User input required:")
                
                preprocessed_file_ready = ques_funcs.provide_choices_question('> You indicated to skip data preprocessing. Load data or quit?', ['yes','quit']) 
                stopwatch.start()
                if preprocessed_file_ready == 'yes':
                    runtime_logger.phase_start(
                        "pickle_load.preprocessed",
                        "Loading preprocessed pickle bundle.",
                        details={"preprocessed_data_dir": preprocessed_data_dir},
                    )
                    loaded_preprocessed_run = load_selected_pickle_bundle_run(
                        reference_prompt='> Please indicate the location of master_structure_reference_dict.',
                        reference_title='Open the master_structure_reference_dict file',
                        info_prompt='> Please indicate the location of master_structure_info_dict.',
                        info_title='Open the master_structure_info_dict file',
                        initialdir=preprocessed_data_dir,
                        output_dir=output_dir,
                    )
                    master_structure_reference_dict = loaded_preprocessed_run.master_structure_reference_dict
                    master_structure_info_dict = loaded_preprocessed_run.master_structure_info_dict
                    specific_output_dir = loaded_preprocessed_run.specific_output_dir
                    raw_mc_output_dir = loaded_preprocessed_run.raw_mc_output_dir
                    runtime_logger.attach_output_dir(specific_output_dir)
                    runtime_logger.phase_end(
                        "pickle_load.preprocessed",
                        "Loaded preprocessed pickle bundle.",
                        details={
                            "reference_dict_path": loaded_preprocessed_run.reference_dict_path_str,
                            "info_dict_path": loaded_preprocessed_run.info_dict_path_str,
                            "specific_output_dir": specific_output_dir,
                        },
                    )
                    runtime_logger.checkpoint(
                        "run_output_dir.ready",
                        "Attached runtime logger to loaded preprocessed run output directory.",
                        details={
                            "specific_output_dir": specific_output_dir,
                            "raw_mc_output_dir": raw_mc_output_dir,
                        },
                    )
                elif preprocessed_file_ready == 'quit':
                    print('> To save a preprocessed dataset to disk, run with preprocessing and pickle data options on.')
                    stopwatch.stop()
                    input("[bold red]Press enter to continue:")
                    stopwatch.start()
                    
                    sys.exit("> You have quit the programme.")
                else:
                    sys.exit("> You have quit the programme.")
                                
                live_display.start()
                important_info.add_text_line("Loaded master_structure_reference_dict from: "+ loaded_preprocessed_run.reference_dict_path_str, live_display)
                important_info.add_text_line("Loaded master_structure_info_dict from: "+ loaded_preprocessed_run.info_dict_path_str, live_display)


                #### REBUILD NON-PICKLABLE OBJECTS

                resolved_frozen_preprocessed_bundle_config = (
                    resolve_loaded_frozen_preprocessed_bundle_config(
                        master_structure_info_dict,
                        pipeline_config.preprocessing.build_frozen_preprocessed_bundle_config(),
                        runtime_logger=runtime_logger,
                    )
                )

                live_display = rebuild_loaded_preprocessed_runtime_objects(
                    master_structure_reference_dict,
                    master_structure_info_dict,
                    structs_referenced_list_generalized,
                    structs_referenced_dict,
                    bx_ref,
                    dose_ref,
                    mr_adc_ref,
                    resolved_frozen_preprocessed_bundle_config,
                    pipeline_config.replay,
                    patients_progress,
                    completed_progress,
                    indeterminate_progress_sub,
                    live_display,
                )

                configure_transform_precompute_settings(
                    master_structure_info_dict,
                    pipeline_config.optimizer.optimizer_v2_search_config,
                    pipeline_config.optimizer.num_stochastic_targeting_transform_samples_input,
                )
                configure_runtime_random_seed_settings(
                    master_structure_info_dict,
                    pipeline_config.random_seeds.transform_generation_random_seed,
                    pipeline_config.random_seeds.optimizer_v1_random_seed,
                )
            ###
            

            specific_output_dir = master_structure_info_dict["Global"]["Specific output dir"]
            raw_mc_output_dir = master_structure_info_dict["Global"]["Raw MC output dir"]

            no_cohort_mr_adc_flag = True
            for patientUID, pydicom_item in master_structure_reference_dict.items():
                if mr_adc_ref in pydicom_item:
                    no_cohort_mr_adc_flag = False
                    break
        

            

            live_display.refresh()
        


            lower_bound_dose_value, live_display = render_processed_dataset_debug_processer(
                master_structure_reference_dict,
                structs_referenced_list,
                structs_referenced_dict,
                bx_ref,
                dose_ref,
                mr_adc_ref,
                plan_ref,
                lower_bound_dose_value,
                show_processed_3d_datasets_renderings,
                show_processed_3d_datasets_renderings_plotly_dict,
                live_display,
            )
            live_display = sampled_biopsy_processing_processer(
                master_structure_reference_dict,
                master_structure_info_dict,
                bx_ref,
                bx_sample_pts_lattice_spacing,
                parallel_pool,
                indeterminate_progress_main,
                patients_progress,
                biopsies_progress,
                completed_progress,
                live_display,
                stopwatch,
                show_reconstructed_biopsy_in_biopsy_coord_sys_tr_and_rot,
            )
            annotate_target_dil_optimizer_v2_outputs_with_biopsy_sampling_audit(
                master_structure_reference_dict,
                bx_ref,
                all_ref_key,
            )

            live_display = biopsy_double_sextant_processer(
                master_structure_reference_dict,
                master_cohort_patient_data_and_dataframes,
                all_ref_key,
                bx_ref,
                oar_ref,
                biopsy_z_voxel_length,
                live_display,
            )

          
            if perform_MC_sim == True:
                max_simulations = master_structure_info_dict["Global"]["MC info"]["Max of num MC simulations"]

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
                






                #live_display.stop()
                # Perform all bx only transformations

                indeterminate_task_bx_only_transforms = indeterminate_progress_main.add_task("[red]Shifting BXs (bx only transforms)", total=None)
                indeterminate_task_bx_only_transforms_completed = completed_progress.add_task("[green]Shifting BXs (bx only transforms)", visible = False, total = 1)
                #indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Shifting BXs (bx only transforms)", total = None)

                # For efficiency debugging
                """
                lp = LineProfiler()
                lp.add_function(MC_prepper_funcs.biopsy_only_transformer)
                lp_wrapper = lp(MC_prepper_funcs.biopsy_only_transformer)

                lp_wrapper(master_structure_reference_dict,
                                                        bx_ref,
                                                        max_simulations,
                                                        simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                                        inspect_self_biopsy_dilate_bool,
                                                        inspect_self_biopsy_dilate_and_rotate_bool,
                                                        inspect_self_biopsy_dilate_and_rotate_and_translate_bool)

                lp.print_stats()

                input("Press Enter to continue...")
                """
                MC_prepper_funcs.biopsy_only_transformer(master_structure_reference_dict,
                                                        bx_ref,
                                                        max_simulations,
                                                        simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                                        inspect_self_biopsy_dilate_bool,
                                                        inspect_self_biopsy_dilate_and_rotate_bool,
                                                        inspect_self_biopsy_dilate_and_rotate_and_translate_bool)
                
                #indeterminate_progress_sub.update(indeterminate_task, visible = False)
                indeterminate_progress_main.update(indeterminate_task_bx_only_transforms, visible = False, refresh = True)
                completed_progress.update(indeterminate_task_bx_only_transforms_completed, advance = 1, visible = True, refresh = True)
                

                # Shift anatomy OLD
                """
                MC_prepper_funcs.biopsy_and_structure_shifter(master_structure_reference_dict,
                                 bx_ref,
                                 structs_referenced_list,
                                 simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                                 max_simulations,
                                 plot_uniform_shifts_to_check_plotly,
                                 plot_shifted_biopsies = plot_shifted_biopsies,
                                 plot_translation_vectors_pointclouds = plot_translation_vectors_pointclouds
                                 )
                """
                

                # Transform biopsy based on relative structures NEW

                indeterminate_task_bx_rel_transforms = indeterminate_progress_main.add_task("[red]Shifting BXs (rel structs transforms)", total=None)
                indeterminate_task_bx_rel_transforms_completed = completed_progress.add_task("[green]Shifting BXs (rel structs transforms)", visible = False, total = 1)
                #indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Shifting BXs (rel structs transforms)", total = None)
                
                # For efficiency debugging
                """
                lp = LineProfiler()
                lp.add_function(MC_prepper_funcs.biopsy_transformer_to_relative_structures)
                lp_wrapper = lp(MC_prepper_funcs.biopsy_transformer_to_relative_structures)

                lp_wrapper(master_structure_reference_dict,
                                 structs_referenced_list,
                                 bx_ref,
                                 num_MC_containment_simulations_input,
                                 inspect_relative_structure_rotate_and_shift_number)

                lp.print_stats()

                input("Press Enter to continue...")
                """
                
                MC_prepper_funcs.biopsy_transformer_to_relative_structures(master_structure_reference_dict,
                                 structs_referenced_list,
                                 bx_ref,
                                 num_MC_containment_simulations_input,
                                 inspect_relative_structure_rotate_and_shift_number
                                 )


                indeterminate_progress_main.update(indeterminate_task_bx_rel_transforms, visible = False, refresh = True)
                completed_progress.update(indeterminate_task_bx_rel_transforms_completed, advance = 1, visible = True, refresh = True)
                #indeterminate_progress_sub.update(indeterminate_task, visible = False)
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
                                                                                            rectum_ref_key,
                                                                                            urethra_ref_key, 
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
                                                                                            show_num_containment_demonstration_plots,
                                                                                            containment_results_structure_types_to_show_per_trial,
                                                                                            show_num_nearest_neighbour_surface_boundary_demonstration,
                                                                                            show_num_relative_structure_centroid_demonstration,
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
                                                                                            raw_data_mc_containment_dump_bool,
                                                                                            keep_light_containment_and_distances_to_relative_structures_dataframe_bool,
                                                                                            show_non_bx_relative_structure_z_dilation_bool,
                                                                                            show_non_bx_relative_structure_xy_dilation_bool,
                                                                                            generate_cuda_log_files_MC_containment_sim,
                                                                                            custom_cuda_kernel_type,
                                                                                            constant_z_slice_polygons_handler_option,
                                                                                            remove_consecutive_duplicate_points_in_polygons,
                                                                                            interp_dist_caps,
                                                                                            cuml_NN_algo,
                                                                                            check_if_end_caps_filled_proper_NN_num,
                                                                                            nn_search_end_cap_grid_factor,
                                                                                            tissue_volume_operator_dictionary
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
                
                live_display.start(refresh=True)
                #live_display.stop()

                

                # copy uncertainty file used for simulation to output folder 
                shutil.copy(uncertainties_file_filled, specific_output_dir)

                if plot_immediately_after_simulation == False:
                    sys.exit('> Programme exited.')

            elif perform_MC_sim == False:
                important_info.add_text_line(
                    "Skipping MC simulation; continuing with current in-memory data only.",
                    live_display,
                )
                live_display.refresh()
                                
    

            rich_preambles.section_completed("Simulations", section_start_time, completed_progress, completed_sections_manager, runtime_logger=runtime_logger)    
            

            section_start_time = datetime.now() 
            runtime_logger.phase_start("section.dataframes_and_directories", "Starting section: Dataframes and directories.")
            
            # BEGIN SECTION TO DO AFTER READING MASTER STRUCTURE AND INFO FILES


            preprocessing_complete_bool = master_structure_info_dict["Global"]["Preprocessing info"]["Preprocessing performed"]
            mc_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC sim performed']
            mc_containment_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC containment sim performed']
            mc_dose_sim_complete_bool = master_structure_info_dict['Global']["MC info"]['MC dose sim performed']
            mc_mr_sim_complete = master_structure_info_dict['Global']["MC info"]['MC MR sim performed']

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

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 2.5", total = None)
                cohort_simulated_biopsy_preparation_dataframe = dataframe_builders.cohort_simulated_biopsy_preparation_dataframe_builder(master_structure_reference_dict,
                                       all_ref_key,
                                       bx_ref)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Simulated biopsy preparation dataframe"] = cohort_simulated_biopsy_preparation_dataframe
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

                # guidance-map firing-depth recommendations dataframe (precomputed outside plotter)
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3.5", total = None)
                precompute_guidance_map_firing_depth_recommendations_for_run(
                    master_structure_reference_dict=master_structure_reference_dict,
                    master_cohort_patient_data_and_dataframes=master_cohort_patient_data_and_dataframes,
                    dil_ref=dil_ref,
                    all_ref_key=all_ref_key,
                    oar_ref=oar_ref,
                    rectum_ref=rectum_ref_key,
                    biopsy_fire_travel_distances=biopsy_fire_travel_distances,
                    biopsy_needle_compartment_length=biopsy_needle_compartment_length,
                    interp_inter_slice_dist=interp_inter_slice_dist,
                    interp_intra_slice_dist=interp_intra_slice_dist,
                    radius_for_normals_estimation=radius_for_normals_estimation,
                    max_nn_for_normals_estimation=max_nn_for_normals_estimation,
                    biopsy_needle_tip_length=biopsy_needle_tip_length,
                    planning_config=guidance_map_planning_config,
                    runtime_logger=runtime_logger,
                )
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                render_guidance_maps_for_run(
                    master_structure_reference_dict,
                    master_structure_info_dict,
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
                    biopsy_needle_tip_length,
                    guidance_map_render_config,
                    important_info=important_info,
                    live_display=live_display,
                    patients_progress=patients_progress,
                    completed_progress=completed_progress,
                    runtime_logger=runtime_logger,
                )

                # structure radiomic 3D segmentation features dataframe
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 4", total = None)
                structure_cohort_3d_radiomic_features_dataframe = dataframe_builders.cohort_structure_features_dataframe_builder(master_structure_reference_dict,
                                                structs_referenced_list,
                                                bx_ref,
                                                all_ref_key)
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

                """
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
                """

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                cohort_all_structure_transformations_pandas_data_frame = dataframe_builders.all_structure_shifts_by_trial_dataframe_builder(master_structure_reference_dict,
                                                    structs_referenced_list,
                                                    bx_ref,
                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: All MC structure transformation values"] = cohort_all_structure_transformations_pandas_data_frame
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

                """
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 3", total = None)
                cohort_mc_tissue_class_pt_wise_results_dataframe = dataframe_builders.cohort_and_multi_biopsy_mc_tissue_class_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: mutual tissue class mc results"] = cohort_mc_tissue_class_pt_wise_results_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                """

                # create global tissue class scores dataframes for each biopsy (deprecated because mutual info superceeded by sum to one)
                """
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 4", total = None)
                cohort_global_tissue_class_by_tissue_type_dataframe = dataframe_builders.global_scores_by_tissue_class_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: tissue class global scores (tissue type)"] = cohort_global_tissue_class_by_tissue_type_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                """

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 5", total = None)
                cohort_global_tissue_class_by_structure_dataframe = dataframe_builders.global_scores_by_specific_structure_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: tissue class global scores (structure)"] = cohort_global_tissue_class_by_structure_dataframe
                annotate_target_dil_optimizer_v2_outputs_with_downstream_mc_scores(
                    master_structure_reference_dict,
                    all_ref_key,
                    num_MC_containment_simulations_input,
                )
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                # All binom est info for cohort (deprecated because mutual info superceeded by sum to one)
                """
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 7", total = None) 
                cohort_all_binom_est_data_by_pt_and_voxel = dataframe_builders.cohort_creator_binom_est_by_pt_and_voxel_dataframe(master_structure_reference_dict,
                                                       bx_ref)  
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Entire point-wise binom est distribution"] = cohort_all_binom_est_data_by_pt_and_voxel
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                """

                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 9", total = None) 
                cohort_mc_distances_global_results_dataframe, cohort_mc_distances_pt_wise_results_dataframe, cohort_mc_distances_voxel_wise_results_dataframe = dataframe_builders.cohort_relative_structure_distances_dataframe_builder(master_structure_reference_dict,
                                                          bx_ref,
                                                          all_ref_key)
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Tissue class - distances global results"] = cohort_mc_distances_global_results_dataframe
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Tissue class - distances pt-wise results"] = cohort_mc_distances_pt_wise_results_dataframe
                master_cohort_patient_data_and_dataframes["Dataframes"]["Cohort: Tissue class - distances voxel-wise results"] = cohort_mc_distances_voxel_wise_results_dataframe
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 10", total = None) 
                dataframe_builders.cohort_containment_results_and_distances_dataframe_builder_light(master_structure_reference_dict,
                                                                     bx_ref,
                                                                     all_ref_key)                                          
                indeterminate_progress_sub.update(indeterminate_task, visible = False)

                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()


                
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
                runtime_logger.phase_start(
                    "dataframe_building",
                    "Starting dataframe generation for MC dosimetry and MR outputs.",
                    details={
                        "mc_dose_sim_complete": mc_dose_sim_complete,
                        "mc_mr_sim_complete": mc_mr_sim_complete,
                    },
                )
                runtime_logger.checkpoint(
                    "dataframe_building.preflight",
                    "Stopping live display before dataframe generation blackout window.",
                )
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
                
                # takes very long time to build, succeeded by dataframe_builders.global_dosimetry_by_biopsy_dataframe_builder_NEW_multiindex_df
                """
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
                """

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



                # DVH metrics new and improved!
                ### WARNING I THINK THESE DVH METRIC CALCULATIONS ARE INCORRECT! LETS JUST NOT CALC THEM IN THIS ALGO AT ALL!
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
                    print("MC, MR simulation complete")
                    csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Generating dataframes (MC, MR)...', total=None)
                    csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Generating dataframes (MC, MR)', total=1, visible = False)

                    mc_sim_mr_adc_arr_str = "MC data: MR ADC vals for each sampled bx pt arr (nominal & all MC trials)"
                    output_mr_adc_dataframe_str = "Point-wise MR ADC output by MC trial number"
                    mr_adc_col_name_str_prefix = "MR ADC"

                    st = time.time()
                    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~DF 1", total = None)
                    dataframe_builders.all_mr_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(master_structure_reference_dict, 
                                                                            bx_ref, 
                                                                            biopsy_z_voxel_length, 
                                                                            mr_adc_ref,
                                                                            mc_sim_mr_adc_arr_str,
                                                                            mr_adc_col_name_str_prefix,
                                                                            output_mr_adc_dataframe_str)
                    indeterminate_progress_sub.update(indeterminate_task, visible = False)
                    et = time.time()
                    duration = et-st
                    print(f"MR DF1: {duration}")

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
                runtime_logger.phase_end("dataframe_building", "Completed dataframe generation.")
                runtime_logger.memory_snapshot(
                    "dataframe_building",
                    "Captured memory snapshot after dataframe generation.",
                )


                



            # CREATE CSV DIRECTORIES ---------------------------


            #live_display.stop()
            # create global csv output folder
            print("Creating CSV output directories...")
            if any([write_preprocessing_data_to_file, write_containment_to_file_ans, write_dose_to_file_ans, write_cohort_data_to_file]):
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
                            #dataframe.to_parquet(dataframe_file_path, compression='snappy')

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
            live_display.stop()
            if mc_csv_output_dir.is_dir():
                runtime_logger.phase_start(
                    "dataframe_export.write_mc",
                    "Writing MC simulation stored dataframes to file.",
                    details={"mc_csv_output_dir": mc_csv_output_dir},
                )
                important_info.add_text_line("Writing MC sim stored dataframes to file.", live_display)
                csv_dataframe_building_indeterminate = indeterminate_progress_main.add_task('[red]Writing MC sim stored dataframes to file...', total=None)
                csv_dataframe_building_indeterminate_completed = completed_progress.add_task('[green]Writing MC sim stored dataframes to file', total=1, visible = False)

                for patientUID, pydicom_item in master_structure_reference_dict.items():
                    patient_sp_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
                    for dataframe_name, dataframe in pydicom_item[all_ref_key]['Multi-structure MC simulation output dataframes dict'].items():
                        if isinstance(dataframe, pandas.DataFrame):
                            if dataframe_name == "Tissue class - containment and distances (light) results":
                                # Use .parquet extension for the Parquet file
                                dataframe_file_name = f"{patientUID}-{dataframe_name}.parquet"
                                dataframe_file_path = patient_sp_csv_dir.joinpath(dataframe_file_name)
                                dataframe.to_parquet(dataframe_file_path, compression='snappy')
                            else:
                                # Keep using .csv for other dataframes
                                dataframe_file_name = f"{patientUID}-{dataframe_name}.csv"
                                dataframe_file_path = patient_sp_csv_dir.joinpath(dataframe_file_name)
                                dataframe.to_csv(dataframe_file_path)

                    for sp_bx in pydicom_item[bx_ref]:
                        bx_name = sp_bx["ROI"]
                        bx_type = sp_bx["Simulated type"]
                        bx_index_number = sp_bx["Index number"] 
                        bx_sp_csv_dir = patient_sp_bx_sp_output_csv_dir_dict[patientUID][bx_index_number]
                        for dataframe_name, dataframe in sp_bx['Output data frames'].items():
                            if isinstance(dataframe, pandas.DataFrame):
                                if (
                                    dataframe_name == "Point-wise dose output by MC trial number"
                                    or dataframe_name == "Point-wise MR ADC output by MC trial number"
                                    or dataframe_name == "Voxel-wise dose output by MC trial number"
                                    or dataframe_name == "Cumulative DVH by MC trial"
                                    or dataframe_name == "Differential DVH by MC trial"
                                ):
                                    # Use .parquet extension for the Parquet file
                                    dataframe_file_name = f"{patientUID}-{bx_type}-{bx_name}-{bx_index_number}-{dataframe_name}.parquet"
                                    dataframe_file_path = bx_sp_csv_dir.joinpath(dataframe_file_name)
                                    dataframe.to_parquet(dataframe_file_path, compression='snappy')
                                else:
                                    dataframe_file_name = f"{patientUID}-{bx_type}-{bx_name}-{bx_index_number}-{dataframe_name}.csv"
                                    dataframe_file_path = bx_sp_csv_dir.joinpath(dataframe_file_name)
                                    dataframe.to_csv(dataframe_file_path)
                                    #dataframe.to_parquet(dataframe_file_path, compression='snappy')

                indeterminate_progress_main.update(csv_dataframe_building_indeterminate, visible = False)
                completed_progress.update(csv_dataframe_building_indeterminate_completed, advance = 1,visible = True)
                live_display.refresh()
                runtime_logger.phase_end(
                    "dataframe_export.write_mc",
                    "Completed writing MC simulation stored dataframes to file.",
                )
                

            if validate_phase3b_in_memory_patient_stitching_bool == True:
                phase3b_validation_output_dir = specific_output_dir.joinpath(
                    "validation",
                    "phase3b_in_memory_stitching",
                )
                if runtime_logger is not None:
                    runtime_logger.phase_start(
                        "phase3b.in_memory_stitch_validation",
                        "Starting Phase 3B in-memory patient-fragment stitch validation.",
                        details={"output_dir": phase3b_validation_output_dir},
                    )
                validation_df, stitched_tables = build_in_memory_stitch_validation(
                    master_structure_reference_dict=master_structure_reference_dict,
                    master_cohort_patient_data_and_dataframes=master_cohort_patient_data_and_dataframes,
                    all_ref_key=all_ref_key,
                    bx_ref=bx_ref,
                    return_stitched_tables=True,
                )
                validation_path, validation_summary_path = write_in_memory_stitch_validation_outputs(
                    validation_df,
                    stitched_tables,
                    phase3b_validation_output_dir,
                    write_stitched_tables=write_phase3b_in_memory_stitched_tables_bool,
                )
                validation_summary = summarize_in_memory_stitch_validation(validation_df)
                important_info.add_text_line(
                    "Phase 3B in-memory stitch validation: {} matches, {} mismatches, {} missing source fragments, {} missing final dataframes.".format(
                        validation_summary["matched_count"],
                        validation_summary["mismatch_count"],
                        validation_summary["missing_source_fragment_count"],
                        validation_summary["missing_final_dataframe_count"],
                    ),
                    live_display,
                )
                if runtime_logger is not None:
                    runtime_logger.phase_end(
                        "phase3b.in_memory_stitch_validation",
                        "Completed Phase 3B in-memory patient-fragment stitch validation.",
                        details={
                            "validation_path": validation_path,
                            "validation_summary_path": validation_summary_path,
                            **validation_summary,
                        },
                    )

            if write_phase3c_patient_fragment_output_surface_bool == True:
                phase3c_output_dir = specific_output_dir.joinpath(
                    "validation",
                    PHASE3C_OUTPUT_DIR_NAME,
                )
                if runtime_logger is not None:
                    runtime_logger.phase_start(
                        "phase3c.patient_fragment_output_surface",
                        "Starting Phase 3C patient-fragment output surface generation.",
                        details={"output_dir": phase3c_output_dir},
                    )
                phase3c_result = write_phase3c_output_surface(
                    master_structure_reference_dict=master_structure_reference_dict,
                    master_cohort_patient_data_and_dataframes=master_cohort_patient_data_and_dataframes,
                    all_ref_key=all_ref_key,
                    bx_ref=bx_ref,
                    output_dir=phase3c_output_dir,
                    write_stitched_tables=write_phase3c_stitched_final_artifacts_bool,
                )
                phase3c_stitch_summary = phase3c_result.stitch_validation_summary
                important_info.add_text_line(
                    "Phase 3C output surface: wrote {} artifacts | stitch validation: {} matches, {} mismatches, {} missing source fragments, {} missing final dataframes.".format(
                        phase3c_result.artifact_count,
                        phase3c_stitch_summary["matched_count"],
                        phase3c_stitch_summary["mismatch_count"],
                        phase3c_stitch_summary["missing_source_fragment_count"],
                        phase3c_stitch_summary["missing_final_dataframe_count"],
                    ),
                    live_display,
                )
                if runtime_logger is not None:
                    runtime_logger.phase_end(
                        "phase3c.patient_fragment_output_surface",
                        "Completed Phase 3C patient-fragment output surface generation.",
                        details={
                            "output_dir": phase3c_result.output_dir,
                            "manifest_path": phase3c_result.manifest_path,
                            "summary_path": phase3c_result.summary_path,
                            "stitch_validation_path": phase3c_result.stitch_validation_path,
                            "stitch_validation_summary_path": phase3c_result.stitch_validation_summary_path,
                            "schema_coverage_path": phase3c_result.schema_coverage_path,
                            "schema_coverage_summary_path": phase3c_result.schema_coverage_summary_path,
                            "schema_unmatched_manifest_path": phase3c_result.schema_unmatched_manifest_path,
                            "schema_data_dictionary_csv_path": phase3c_result.schema_data_dictionary_csv_path,
                            "schema_data_dictionary_markdown_path": phase3c_result.schema_data_dictionary_markdown_path,
                            **phase3c_result.summary,
                        },
                    )

            if PatientRunnerMainValidationMode(patient_runner_validation_mode) != PatientRunnerMainValidationMode.DISABLED:
                patient_runner_validation_output_dir = specific_output_dir.joinpath(
                    "validation",
                    DEFAULT_PATIENT_RUNNER_SHADOW_OUTPUT_DIR_NAME,
                )
                if runtime_logger is not None:
                    runtime_logger.phase_start(
                        "patient_runner.shadow_output_validation",
                        "Starting patient-runner shadow output validation.",
                        details={
                            "mode": patient_runner_validation_mode,
                            "output_dir": patient_runner_validation_output_dir,
                        },
                    )
                patient_runner_validation_result = run_patient_runner_main_validation(
                    master_structure_reference_dict=master_structure_reference_dict,
                    master_structure_info_dict=master_structure_info_dict,
                    master_cohort_patient_data_and_dataframes=master_cohort_patient_data_and_dataframes,
                    legacy_keys=LegacyRuntimeKeys(
                        all_ref_key=all_ref_key,
                        bx_ref=bx_ref,
                        by_patient_key=by_patient_key,
                        global_key=global_key,
                        global_num_cases_key=global_num_cases_key,
                    ),
                    output_root=specific_output_dir,
                    config=PatientRunnerMainValidationConfig(
                        mode=patient_runner_validation_mode,
                        patient_uids=patient_runner_validation_patient_uids,
                        final_table_names=patient_runner_validation_final_table_names,
                        source_table_names=patient_runner_validation_source_table_names,
                        output_dir=patient_runner_validation_output_dir,
                        write_outputs=patient_runner_validation_write_outputs_bool,
                        write_assembled_tables=patient_runner_validation_write_assembled_tables_bool,
                    ),
                )
                patient_runner_validation_summary = summarize_patient_runner_main_validation(
                    patient_runner_validation_result,
                )
                patient_runner_cohort_validation_summary = patient_runner_validation_summary.get(
                    "validation_summary",
                    {},
                ) or {}
                important_info.add_text_line(
                    "Patient-runner shadow output validation: {} matches, {} mismatches, {} missing assembled tables, {} missing final dataframes.".format(
                        patient_runner_cohort_validation_summary.get("matched_count", 0),
                        patient_runner_cohort_validation_summary.get("mismatch_count", 0),
                        patient_runner_cohort_validation_summary.get("missing_assembled_table_count", 0),
                        patient_runner_cohort_validation_summary.get("missing_final_dataframe_count", 0),
                    ),
                    live_display,
                )
                if runtime_logger is not None:
                    runtime_logger.phase_end(
                        "patient_runner.shadow_output_validation",
                        "Completed patient-runner shadow output validation.",
                        details=patient_runner_validation_summary,
                    )


            # cohort 
            if write_cohort_data_to_file == True:
                important_info.add_text_line("Writing cohort CSVs to file.", live_display)

                validation_only_cohort_dataframe_names = {
                    "Cohort: Simulated biopsy planned vs realized centroid variation validation",
                }

                for dataframe_name, dataframe in master_cohort_patient_data_and_dataframes['Dataframes'].items():
                    if dataframe_name in validation_only_cohort_dataframe_names:
                        continue
                    if isinstance(dataframe, pandas.DataFrame):

                        dataframe_file_name = str(dataframe_name)+ '.csv'
                        dataframe_file_path = cohort_csv_output_dir.joinpath(dataframe_file_name)
                        dataframe.to_csv(dataframe_file_path)
                        #dataframe.to_parquet(dataframe_file_path, compression='snappy')

    
            
            runtime_logger.checkpoint(
                "production_plots.skipped",
                "Legacy production-plot orchestration is disabled in main; use a dedicated workflow when needed.",
            )
            rich_preambles.section_completed("Legacy production plots (skipped)", section_start_time, completed_progress, completed_sections_manager, runtime_logger=runtime_logger)

            live_display.stop()
    if runtime_logger is not None:
        runtime_logger.mark_completed("Programme complete.")
    sys.exit("> Programme complete.")


def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def structure_referencer(data_removals_dict_bx,
                        data_removals_dict_prostate,
                        data_removals_dict_dil,
                        data_removals_dict_urethra,
                        data_removals_dict_rectum,
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
                         structs_referenced_dict,
                         ds_ref,
                         pln_ref,
                         mr_adc_ref,
                         mr_t2_ref,
                         us_ref,
                         all_ref_key,
                         mr_global_multi_structure_output_dataframe_str,
                         mr_global_by_voxel_multi_structure_output_dataframe_str,
                         bx_sim_locations_dict,
                         rectum_list,
                         urethra_list,
                         interp_inter_slice_dist,
                         interp_intra_slice_dist,
                         simulated_biopsy_fraction_numbers_to_create,
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

            ### Remove unwanted data (prostate)
            if UID in data_removals_dict_prostate.keys():
                for prost_id_to_remove in data_removals_dict_prostate[UID]:
                    for prost in filtered_OARs:
                        if prost.ROIName == prost_id_to_remove:
                            filtered_OARs.remove(prost)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Prostate: {prost_id_to_remove})) ", live_display)

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

            ### Remove unwanted data (dils)
            if UID in data_removals_dict_dil.keys():
                for dil_id_to_remove in data_removals_dict_dil[UID]:
                    for dil in filtered_DILs:
                        if dil.ROIName == dil_id_to_remove:
                            filtered_DILs.remove(dil)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, DIL: {dil_id_to_remove})) ", live_display)

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

            ### Remove unwanted data (rectum)
            if UID in data_removals_dict_rectum.keys():
                for rect_id_to_remove in data_removals_dict_rectum[UID]:
                    for rect in filtered_rectums:
                        if rect.ROIName == rect_id_to_remove:
                            filtered_rectums.remove(rect)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Rect: {rect_id_to_remove})) ", live_display)

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

            ### Remove unwanted data (urethra)
            if UID in data_removals_dict_urethra.keys():
                for uret_id_to_remove in data_removals_dict_urethra[UID]:
                    for uret in filtered_urethras:
                        if uret.ROIName == uret_id_to_remove:
                            filtered_urethras.remove(uret)
                            important_info.add_text_line(f"Removed data-point (Pt: {UID}, Uret: {uret_id_to_remove})) ", live_display)

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

            ### Remove unwanted data (bx)
            if UID in data_removals_dict_bx.keys():
                for bx_id_to_remove in data_removals_dict_bx[UID]:
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
                         "MC data: MC sim compiled distances global dataframe": None,
                         "MC data: MC sim compiled distances point-wise dataframe": None,
                         "MC data: MC sim compiled distances voxel-wise dataframe": None,
                         "MC data: MC sim containment and distance all trials dataframe (light)": None,
                         "MC data: compiled sim results dataframe": None,
                         "MC data: compiled sim sum-to-one results dataframe": None,
                         #"MC data: mutual compiled sim results dataframe": None,
                         "MC data: compiled sim results": None, 
                         "MC data: mutual compiled sim results": None,
                         #"MC data: tumor tissue probability": None,
                         #"MC data: miss structure tissue probability": None,
                         #"MC data: tissue length above threshold dict": None,
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
                         "Output csv file paths dict": {}, 
                         "Output data frames": {"Dose output Z and radius": None,
                                                "Dose output voxelized": None,
                                                "Point-wise dose output by MC trial number": None,
                                                "Voxel-wise dose output by MC trial number": None,
                                                #"Mutual containment output by bx point": None,
                                                "Differential DVH by MC trial": None,
                                                "Cumulative DVH by MC trial": None},
                         "Output dicts for data frames": {},  
                         "KDtree": None, 
                         "Nearest neighbours objects": []
                         } for idx, x in enumerate(filtered_BXs)]


            


            bpsy_ref_index_start = len(bpsy_ref)
            patient_fraction_number = misc_tools.extract_number_from_string(str(structure_item[0x0010,0x0020].value), fraction_prefixes)
            if simulated_biopsy_fraction_numbers_to_create == 'all':
                create_simulated_biopsies_for_this_fraction = True
            elif type(simulated_biopsy_fraction_numbers_to_create) in [list, tuple, set]:
                create_simulated_biopsies_for_this_fraction = patient_fraction_number in simulated_biopsy_fraction_numbers_to_create
            else:
                create_simulated_biopsies_for_this_fraction = patient_fraction_number == simulated_biopsy_fraction_numbers_to_create
            sim_bpsy_ref_index_start = 0
            bpsy_ref_simulated_total = []
            for bx_sim_type_str, bx_sim_type_dict in bx_sim_locations_dict.items():
                if bx_sim_type_dict["Create"] == True and create_simulated_biopsies_for_this_fraction == True:
                    sim_bx_relative_to = bx_sim_type_dict["Relative to struct type"]
                    bx_sim_ref_identifier_str = bx_sim_type_dict["Identifier string"]

                    sim_bx_relative_to_contour_names = structs_referenced_dict[sim_bx_relative_to]["Contour names"]
                    

                    # Determine the appropriate removal list based on the relative structure type.
                    if bx_sim_type_dict["Relative to struct type"] == st_ref_list[2]:
                        removal_list = data_removals_dict_dil.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[0]:
                        removal_list = data_removals_dict_bx.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[1]:
                        removal_list = data_removals_dict_prostate.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[3]:
                        removal_list = data_removals_dict_rectum.get(UID, [])
                    elif bx_sim_type_dict["Relative to struct type"] == st_ref_list[4]:
                        removal_list = data_removals_dict_urethra.get(UID, [])
                    else:
                        removal_list = []

                    # Now filter simulated biopsy candidates by including the removal condition.
                    filtered_sim_BXs = [
                        x for x in structure_item.StructureSetROISequence
                        if any(contour.lower() in x.ROIName.lower() for contour in sim_bx_relative_to_contour_names)
                        and x.ROIName not in removal_list
                    ]

                    # old doesnt account for data point removals of relative structures for simulating biopsies!
                    #filtered_sim_BXs = [x for x in structure_item.StructureSetROISequence if sim_bx_relative_to.lower() in x.ROIName.lower()]
                    
                    
                    bpsy_ref_simulated = [{"ROI": "Bx_Tr_"+bx_sim_ref_identifier_str+" " + x.ROIName, 
                                "Ref #": bx_sim_ref_identifier_str +" "+ x.ROIName,
                                "Index number": bpsy_ref_index_start + sim_bpsy_ref_index_start + idx,
                                "Struct type": st_ref_list[0],
                                "Simulated bool": True,
                                "Simulated type": bx_sim_type_str,
                                "Transport family": bx_sim_type_dict.get("Transport family", "identity"),
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
                                "MC data: MC sim compiled distances global dataframe": None,
                                "MC data: MC sim compiled distances point-wise dataframe": None,
                                "MC data: MC sim compiled distances voxel-wise dataframe": None,
                                "MC data: MC sim containment and distance all trials dataframe (light)": None,
                                "MC data: compiled sim results dataframe": None,
                                "MC data: compiled sim sum-to-one results dataframe": None,
                                #"MC data: mutual compiled sim results dataframe": None,
                                "MC data: compiled sim results": None,
                                "MC data: mutual compiled sim results": None, 
                                #"MC data: tumor tissue probability": None,
                                #"MC data: miss structure tissue probability": None,
                                #"MC data: tissue length above threshold dict": None,
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
                                "Simulated biopsy transport request dict": None,
                                "Output csv file paths dict": {}, 
                                "Output data frames": {"Dose output Z and radius": None,
                                                       "Dose output voxelized": None,
                                                       "Point-wise dose output by MC trial number": None,
                                                       "Voxel-wise dose output by MC trial number": None,
                                                       #"Mutual containment output by bx point": None,
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
                                                                                  "Simulated biopsy preparation dataframe": None,
                                                                                  #"Structure information dimension": pandas.DataFrame(),
                                                                                  #"Structure information (Non-BX)": pandas.DataFrame(),
                                                                                  "Nearest DILs info dataframe": None,
                                                                                  "Biopsy optimization - Cumulative projection (all points within prostate) dataframe": None,
                                                                                  "Biopsy optimization - DIL centroids optimal targeting dataframe": None,
                                                                                  "Biopsy optimization - Optimal DIL targeting dataframe": None,
                                                                                  "Biopsy optimization - Optimal DIL targeting entire lattice dataframe": None,
                                                                                  TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY: None,
                                                                                  TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY: None,
                                                                                  "Biopsy optimization - Guidance-map firing depth recommendations dataframe": None,
                                                                                  "3D radiomic features all OAR and DIL structures": None,
                                                                                  "Per sample point prostate double sextant classification": None,
                                                                                  "Per voxel prostate double sextant classification": None,
                                                                                  "Simulated biopsy planned vs realized centroid variation validation": None,
                                                                                  "Prostate only points MR ADC dataframe (temporary for pre-processing)": None,
                                                                                  "MR - ADC - summary statistics by structure dataframe": None},    
                        "Multi-structure MC simulation output dataframes dict": {"All MC structure transformation values": None,
                                                                                 "Tissue class - Global tissue class statistics": None,
                                                                                 "Tissue class - Global tissue by structure statistics": None,
                                                                                 "Tissue class - Tissue length above threshold": None,
                                                                                 "Tissue class - sum-to-one mc results": None,
                                                                                 "Tissue class - distances global results": None,
                                                                                 "Tissue class - distances pt-wise results": None,
                                                                                 "Tissue class - distances voxel-wise results": None,
                                                                                 "Tissue class - containment and distances (light) results": None,
                                                                                 #"Tissue class - Pt wise mutual tissue class results": None,
                                                                                 "Tissue class - Pt wise structure specific results": None,
                                                                                 #"Dosimetry - All points and trials": pandas.DataFrame(),
                                                                                 "DVH metrics (Dx, Vx) statistics": None,
                                                                                 "MR - " + str(mr_global_multi_structure_output_dataframe_str): None,
                                                                                 "MR - " + str(mr_global_by_voxel_multi_structure_output_dataframe_str): None},
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
    
    """
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

    """

    ### MR ADC, updated to account for missing RealWorldValueMappingSequence
    for UID, mr_adc_item_paths_list in MR_ADC_dcms_dict.items():
        mr_adc_ref_dict = {}
        for mr_adc_item_path in mr_adc_item_paths_list:
            with pydicom.dcmread(mr_adc_item_path, defer_size='2 MB') as mr_adc_item:
                seriesinstanceUID = mr_adc_item.SeriesInstanceUID
                mr_adc_ID = UID + mr_adc_item.StudyDate

                rwvm = getattr(mr_adc_item, "RealWorldValueMappingSequence", None)

                # Handle multiple RWVMs
                if rwvm and len(rwvm) > 1:
                    important_info.add_text_line(
                        f"Multiple real world value mappings detected for ({UID}, {mr_adc_ID})", live_display
                    )

                # Safely extract or assign defaults
                if rwvm and len(rwvm) > 0:
                    rwv = rwvm[0]
                    units = str(getattr(rwv.MeasurementUnitsCodeSequence[0], "CodeMeaning", "unknown"))
                    slope = np.array(rwv.RealWorldValueSlope)
                    intercept = np.array(rwv.RealWorldValueIntercept)
                    rwv_units = getattr(rwv, "LUTLabel", "unknown")
                else:
                    units = "mm\u00B2/s (assumed)"
                    slope = np.array([1e-6])  # Default slope for mm2/s
                    intercept = np.array([0.0])
                    rwv_units = "mm\u00B2/s (assumed)"
                    important_info.add_text_line(
                        f"No RealWorldValueMappingSequence found for ({UID}, {mr_adc_ID}) – using defaults.",
                        live_display
                    )

                if seriesinstanceUID not in mr_adc_ref_dict:
                    mr_adc_ref_subdict = {
                        "MR ADC ID": mr_adc_ID,
                        "Series instance UID": seriesinstanceUID,
                        "Study date": mr_adc_item.StudyDate,
                        "Pixel arr (all slices)": mr_adc_item.pixel_array,
                        "Pixel spacing": np.array(mr_adc_item.PixelSpacing),
                        "Units": units,
                        "RWVSlope (all slices)": slope,
                        "RWVIntercept (all slices)": intercept,
                        "RWV Units": rwv_units,
                        "Slice thickness": getattr(mr_adc_item, "SliceThickness", -1),
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
                    mr_adc_ref_subdict["Pixel arr (all slices)"] = np.dstack(
                        (mr_adc_ref_subdict["Pixel arr (all slices)"], mr_adc_item.pixel_array)
                    )
                    mr_adc_ref_subdict["RWVSlope (all slices)"] = np.hstack(
                        (mr_adc_ref_subdict["RWVSlope (all slices)"], slope)
                    )
                    mr_adc_ref_subdict["RWVIntercept (all slices)"] = np.hstack(
                        (mr_adc_ref_subdict["RWVIntercept (all slices)"], intercept)
                    )
                    mr_adc_ref_subdict["Image position patient (all slices)"] = np.vstack(
                        (mr_adc_ref_subdict["Image position patient (all slices)"], np.array(mr_adc_item.ImagePositionPatient))
                    )
                    mr_adc_ref_dict[seriesinstanceUID] = mr_adc_ref_subdict

                pass
        pass

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
               "Num optimizer v2 transform samples": None,
               "Num stochastic targeting transform samples": None,
               "Num sample pts per BX core": None, 
               "BX sample pt lattice spacing (mm)": None,
               "BX sample pt volume element (mm^3)": None,
               "Max of num MC simulations": None,
               "Max of generated transform samples": None,
               'MC sim performed': False,
               'MC containment sim performed': False,
               'MC dose sim performed': False,
               'MC MR sim performed': False,
               }

    random_info = {"Transform generation random seed": None,
                   "Optimizer v1 random seed": None,
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
                                               "Random info": random_info,
                                               'Patient specific guidance map figures directory dict': None,
                                               'Guidance map figures dir': None,
                                               "Specific output dir": None
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
    def fill_means_and_sigmas(self, means_arr, sigmas_arr, means_arr_dilations, sigmas_arr_dilations, means_arr_rotations, sigmas_arr_rotations):
        self.uncertainty_data_mean_arr = means_arr
        self.uncertainty_data_sigma_arr = sigmas_arr
        self.uncertainty_data_dilations_mean_arr = means_arr_dilations
        self.uncertainty_data_dilations_sigma_arr = sigmas_arr_dilations
        self.uncertainty_data_rotations_mean_arr = means_arr_rotations
        self.uncertainty_data_rotations_sigma_arr = sigmas_arr_rotations



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








if __name__ == '__main__':    
    main()
    
