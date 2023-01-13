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
    spinner_type = 'christmas'
    
    # The figure dictionary to be plotted, this needs to be requested of the user later in the programme, after the  dicoms are read

    # First we access the data directory, it must be in a location 
    # two levels up from this file
    data_dir = pathlib.Path(__file__).parents[2].joinpath(data_folder_name)
    uncertainty_dir = data_dir.joinpath(uncertainty_folder_name)
    dicom_paths_list = list(pathlib.Path(data_dir).glob("**/*.dcm")) # list all file paths found in the data folder that have the .dcm extension
    dicom_elems_list = list(map(pydicom.dcmread,dicom_paths_list)) # read all the found dicom file paths using pydicom to create a list of FileDataset instances 

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
    
    master_structure_reference_dict, master_structure_info_dict, structs_referenced_list, dose_ref = structure_referencer(RTst_dcms_dict, RTdose_dcms_dict, oaroi_contour_names,dil_contour_names,biopsy_contour_names)


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
    test_ind = 0
    cpu_count = os.cpu_count()
    with multiprocess.Pool(cpu_count) as parallel_pool:
        #st = time.time()
        
        
        with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TextColumn("[green]Patient:"),
                rich.progress.MofNCompleteColumn(),
                rich.progress.TimeElapsedColumn()) as progress:
            patientUID_default = "Initializing"
            processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID_default)
            processing_patients_dose_task = progress.add_task(processing_patients_dose_task_main_description, total=num_patients)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                processing_patients_dose_task_main_description = "[red]Building dose grids [{}]...".format(patientUID)
                progress.update(processing_patients_dose_task, description=processing_patients_dose_task_main_description)

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

                plotting_funcs.plot_geometries(dose_point_cloud)

                progress.update(processing_patients_dose_task, advance=1)
                
                    




        with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TextColumn("[green]Patient:"),
                rich.progress.MofNCompleteColumn(),
                rich.progress.TimeElapsedColumn()) as progress_patient:
            with Progress(rich.progress.SpinnerColumn(spinner_type),
            *Progress.get_default_columns(),
            rich.progress.TextColumn("[green]Structure:"),
            rich.progress.MofNCompleteColumn(),
            rich.progress.TimeElapsedColumn()) as progress_structure:

                patientUID_default = "Initializing"
                processing_patients_task_main_description = "[red]Processing patient structure data [{}]...".format(patientUID_default)
                processing_patients_task = progress.add_task(processing_patients_task_main_description, total=num_patients)

                structureID_default = "Initializing"
                processing_structures_task_main_description = "[green]Processing structures [{}]...".format(structureID_default)
                processing_structures_task = progress.add_task(processing_structures_task_main_description, total=num_general_structs)
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    processing_patients_task_main_description = "[red]Processing patient structure data [{}]...".format(patientUID)
                    progress.update(processing_patients_task, description = processing_patients_task_main_description)
                    for structs in structs_referenced_list:
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                            structureID = specific_structure["ROI"]
                            processing_structures_task_main_description = "[green]Processing structures [{}]...".format(structureID)
                            progress.update(processing_structures_task_main_description, description = processing_structures_task_main_description)

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

                            # plot points with order labels of interpolated intraslice ?
                            #plotting_funcs.point_cloud_with_order_labels(threeDdata_array_fully_interpolated)

                            # plot points with order labels of raw data ?
                            #if test_ind > 1:
                            #    plotting_funcs.point_cloud_with_order_labels(threeDdata_array)
                            #test_ind = test_ind + 1

                            
                            
                            # plot fully interpolated points of z data ?
                            #plotting_funcs.point_cloud_with_order_labels(threeDdata_array_interslice_interpolation)
                            #plotting_funcs.plot_point_clouds(threeDdata_array_interslice_interpolation,threeDdata_array,threeDdata_array_fully_interpolated, label='Unknown')
                            

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
                                drawn_biopsy_array_transpose = biopsy_creator.biopsy_points_creater_by_transport(list_travel_vec,list_centroid_line_first_point,num_centroid_samples_of_centroid_line,np.linalg.norm(travel_vec),False)
                                drawn_biopsy_array = drawn_biopsy_array_transpose.T
                                reconstructed_bx_pcd_color = np.random.uniform(0, 0.7, size=3)
                                reconstructed_biopsy_point_cloud = point_containment_tools.create_point_cloud(drawn_biopsy_array, reconstructed_bx_pcd_color)
                                reconstructed_bx_delaunay_global_convex_structure_obj = point_containment_tools.delaunay_obj(drawn_biopsy_array, reconstructed_bx_pcd_color)
                                reconstructed_bx_delaunay_global_convex_structure_obj.generate_lineset()
                                #plot reconstructions?
                                #plotting_funcs.plot_geometries(reconstructed_biopsy_point_cloud)
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
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure pts arr"] = drawn_biopsy_array
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure point cloud"] = reconstructed_biopsy_point_cloud
                                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure delaunay global"] = reconstructed_bx_delaunay_global_convex_structure_obj


                            progress.update(processing_structures_task, advance=1)
                    progress.update(processing_patients_task, advance=1)


        
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            dose_ref_dict = pydicom_item[dose_ref]
            dose_grid_pcd = dose_ref_dict["Dose grid point cloud"]
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
        num_sample_pts_per_bx = 9
        master_structure_info_dict["Global"]["MC info"]["Num sample pts per BX core"] = num_sample_pts_per_bx
        with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TimeElapsedColumn()) as progress:
            processing_biopsies_task = progress.add_task("[red]Preparing for parallel computing...", total=num_biopsies)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                structs = structs_referenced_list[0]
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                    reconstructed_biopsy_point_cloud = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure point cloud"]
                    reconstructed_biopsy_arr = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure pts arr"]
                    reconstructed_delaunay_global_convex_structure_obj = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure delaunay global"]
                    args_list.append((num_sample_pts_per_bx, reconstructed_delaunay_global_convex_structure_obj.delaunay_triangulation, reconstructed_biopsy_arr, patientUID, structs, specific_structure_index))
                    progress.update(processing_biopsies_task, advance=1)


        #et = time.time()
        #elapsed_time = et - st
        #print('\n Execution time (NON PARALLEL):', elapsed_time, 'seconds')
        
        
        #st = time.time()
        
        with Progress(rich.progress.SpinnerColumn(spinner_type),
                *Progress.get_default_columns(),
                rich.progress.TimeElapsedColumn()) as progress:
            sampling_points_task = progress.add_task("[green]Sampling biopsy points (parallel)...", total=None)
            parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr = parallel_pool.starmap(MC_simulator_convex.point_sampler_from_global_delaunay_convex_structure_parallel, args_list)
        
        #et = time.time()
        #elapsed_time = et - st
        #print('\n Execution time (PARALLEL):', elapsed_time, 'seconds')

        for sampled_bx_pts_arr, axis_aligned_bounding_box_arr, structure_info_dict in parallel_results_sampled_bx_points_from_global_delaunay_arr_and_bounding_box_arr:
            sampled_bx_points_from_global_delaunay_point_cloud_color = np.random.uniform(0, 0.7, size=3)
            sampled_bx_points_from_global_delaunay_point_cloud = point_containment_tools.create_point_cloud(sampled_bx_pts_arr, sampled_bx_points_from_global_delaunay_point_cloud_color)
            axis_aligned_bounding_box = o3d.geometry.AxisAlignedBoundingBox()
            axis_aligned_bounding_box_o3d3dvector_points = o3d.utility.Vector3dVector(axis_aligned_bounding_box_arr)
            axis_aligned_bounding_box.create_from_points(axis_aligned_bounding_box_o3d3dvector_points)
            #axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
            bounding_box_color = np.array([0,0,0], dtype=float)
            axis_aligned_bounding_box.color = bounding_box_color

            # update master dict 
            temp_patient_UID = structure_info_dict["Patient UID"]
            temp_structure_type = structure_info_dict["Structure type"]
            temp_specific_structure_index = structure_info_dict["Specific structure index"]
            master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Random uniformly sampled volume pts arr"] = sampled_bx_pts_arr
            master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Random uniformly sampled volume pts pcd"] = sampled_bx_points_from_global_delaunay_point_cloud
            master_structure_reference_dict[temp_patient_UID][temp_structure_type][temp_specific_structure_index]["Bounding box for random uniformly sampled volume pts"] = axis_aligned_bounding_box

             
            plotting_funcs.plot_geometries(sampled_bx_points_from_global_delaunay_point_cloud,axis_aligned_bounding_box)


        ## begin simulation section
        created_dir = False
        while created_dir == False:
            print('Must create an uncertainties folder at ', uncertainty_dir, '. If the folder already exists it will not be overwritten.')
            uncertainty_dir_generate = ques_funcs.ask_ok('Continue?')

            if uncertainty_dir_generate == True:
                if os.path.isdir(uncertainty_dir) == True:
                    print('Directory already exists')
                    created_dir = True
                else:
                    os.mkdir(uncertainty_dir)
                    print('Directory: ', uncertainty_dir, ' created.')
                    created_dir = True
            else:
                exit_programme = ques_funcs.ask_ok('This directory must be created. Do you want to exit the programme?' )
                if exit_programme == True:
                    sys.exit('Programme exited.')
                else: 
                    pass

        uncertainty_template_generate = ques_funcs.ask_ok('Do you want to generate an uncertainty file template for this patient data repo?')
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
            uncertainty_file_ready = ques_funcs.ask_ok('Is the uncertainty file prepared/filled out?') 
            if uncertainty_file_ready == True:
                print('Please select the file with the dialog box')
                root = tk.Tk() # these two lines are to get rid of errant tkinter window
                root.withdraw() # these two lines are to get rid of errant tkinter window
                # this is a user defined quantity, should be a tab delimited csv file in the future, mu sigma for each uncertainty direction
                uncertainties_file_filled = fd.askopenfilename(title='Open the uncertainties data file', initialdir=data_dir, filetypes=[("Excel files", ".xlsx .xls .csv")])
                with open(uncertainties_file_filled, "r", newline='\n') as uncertainties_file_filled_csv:
                    uncertainties_filled = uncertainties_file_filled_csv
                pandas_read_uncertainties = pandas.read_csv(uncertainties_file_filled, names = [0, 1, 2, 3, 4, 5])  
                print(pandas_read_uncertainties)
            else:
                print('Please fill out the generated uncertainties file generated at ', uncertainties_file)
                ask_to_quit = ques_funcs.ask_ok('Would you like to quit the programme instead?')
                if ask_to_quit == True:
                    sys.exit("You have quit the programme.")
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

        simulation_ans = ques_funcs.ask_ok('Uncertainty data collected. Begin Monte Carlo simulation?')
        
        num_simulations = 11
        master_structure_info_dict["Global"]["MC info"]["Num MC simulations"] = num_simulations
        if simulation_ans ==  True:
            print('Beginning simulation')
            master_structure_reference_dict = MC_simulator_convex.simulator_parallel(parallel_pool, master_structure_reference_dict, structs_referenced_list, dose_ref, master_structure_info_dict, spinner_type)
            #master_structure_reference_dict_simulated = MC_simulator_convex.simulator(master_structure_reference_dict, structs_referenced_list,num_simulations, pandas_read_uncertainties)
            print('test')
        else: 
            pass

    print('Programme has ended.')

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
        bpsy_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Equal num zslice contour pts": None, "Intra-slice interpolation information": None, "Inter-slice interpolation information": None, "Point cloud raw": None, "Delaunay triangulation global structure": None, "Delaunay triangulation zslice-wise list": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts arr": None, "Reconstructed structure point cloud": None, "Reconstructed structure delaunay global": None, "Random uniformly sampled volume pts arr": None, "Random uniformly sampled volume pts pcd": None, "Bounding box for random uniformly sampled volume pts": None, "Uncertainty data": None, "MC data: Generated normal dist random samples arr": None, "MC data: bx only shifted 3darr": None, "MC data: bx and structure shifted dict": None, "MC data: MC sim translation results dict": None, "MC data: compiled sim results": None, "MC data: bx to dose NN search objects list": None, "MC data: Dose NN child obj for each sampled bx pt list": None, "MC data: Dose vals for each sampled bx pt list": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
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
        dose_ref_dict = {"Dose ID": dose_ID, "Study date": dose_item.StudyDate, "Dose pixel data": dose_item.PixelData, "Dose pixel arr": dose_item.pixel_array, "Pixel spacing": [float(item) for item in dose_item.PixelSpacing], "Dose grid scaling": float(dose_item.DoseGridScaling), "Dose units": dose_item.DoseUnits, "Dose type": dose_item.DoseType, "Grid frame offset vector": [float(item) for item in dose_item.GridFrameOffsetVector], "Image orientation patient": [float(item) for item in dose_item.ImageOrientationPatient], "Image position patient": [float(item) for item in dose_item.ImagePositionPatient], "Dose phys space and pixel 3d arr": None, "Dose grid point cloud": None, "KDtree": None}
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
    