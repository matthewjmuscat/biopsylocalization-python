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
import alphashape



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
    # The following could be user input, for now they are defined here, and used throughout 
    # the programme for generality
    Data_folder_name = 'Data'
    modality_list = ['RTSTRUCT','RTDOSE','RTPLAN']
    OAROI_contour_names = ['Prostate','Urethra','Rectum','random']
    Biopsy_contour_names = ['Bx']
    DIL_contour_names = ['DIL']
    
    # The figure dictionary to be plotted, this needs to be requested of the user later in the programme, after the  dicoms are read

    # First we access the data directory, it must be in a location 
    # two levels up from this file
    data_dir = pathlib.Path(__file__).parents[2].joinpath(Data_folder_name)
    dicom_paths_list = list(pathlib.Path(data_dir).glob("**/*.dcm")) # list all file paths found in the data folder that have the .dcm extension
    dicom_elems_list = list(map(pydicom.dcmread,dicom_paths_list)) # read all the found dicom file paths using pydicom to create a list of FileDataset instances 

    # The 0x0008,0x0060 dcm tag specifies the 'Modality', here it is used to identify the type
    # of dicom file 
    RTst_dcms = [x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[0]]
    RTdose_dcms = [x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[1]]
    RTplan_dcms = [x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[2]]
    
    # the below is the first use of the UID_generator(pydicom_obj) function, which is used for the
    # creation of the PatientUID, that is generally created from or referenced from here 
    # throughout the programme, it is formed as "patientname (patientID)"
    RTst_dcms_dict = {UID_generator(x): x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[0]}

    
    master_structure_reference_dict, structs_referenced_list = structure_referencer(RTst_dcms_dict, OAROI_contour_names,DIL_contour_names,Biopsy_contour_names)

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
    num_patients = len(master_structure_reference_dict)
    num_general_structs_per_patient = len(structs_referenced_list)
    num_general_structs = num_patients*num_general_structs_per_patient

    st = time.time()
    with loading_tools.Loader(num_general_structs,"Processing data...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            for structs in structs_referenced_list:
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                    # The below print lines were just for my own understanding of how to access the data structure
                    #print(specific_structure["ROI"])
                    #print(specific_structure["Ref #"])
                    #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                    #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                    # can uncomment surrounding lines to time this particular process
                    #st = time.time()
                    
                    total_structure_points = sum([len(x.ContourData)/3 for x in RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:]])
                    if total_structure_points.is_integer():
                        total_structure_points = int(total_structure_points)
                    else: 
                        raise Exception("Seems the cumulative number of spatial components of contour points is not divisible by three!") 
                    threeDdata_array = np.empty([total_structure_points,3])
                    
                    structure_centroids_array = np.empty([len(RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:]),3])
                    lower_bound_index = 0
                    for index, slice_object in enumerate(RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:]):
                        contour_slice_points = slice_object.ContourData                       
                        threeDdata_zslice = np.fromiter([contour_slice_points[i:i + 3] for i in range(0, len(contour_slice_points), 3)], dtype=np.dtype((np.float64, (3,))))
                        
                        current_zslice_num_points = np.size(threeDdata_zslice,0)
                        threeDdata_array[lower_bound_index:lower_bound_index + current_zslice_num_points] = threeDdata_zslice
                        lower_bound_index = lower_bound_index + current_zslice_num_points 
                        
                        structure_zslice_centroid = np.mean(threeDdata_zslice,axis=0)
                        structure_centroids_array[index] = structure_zslice_centroid
                        
                    #et = time.time()
                    #elapsed_time = et - st
                    #print('\n Execution time:', elapsed_time, 'seconds')

                    master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(threeDdata_array)
                    #pcd_color = np.ndarray((3,1), dtype=np.float64)
                    #pcd_color[:] = 0.
                    pcd_color = np.random.uniform(0, 0.7, size=3)
                    point_cloud.paint_uniform_color(pcd_color)
                    #point_cloud.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(len(np.asarray(pcd.points)), 3)))

                    #delaunay_triangulation = scipy.spatial.Delaunay(threeDdata_array)
                    delaunay_triangulation_obj = delaunay_obj(threeDdata_array, pcd_color)
                    master_structure_reference_dict[patientUID][structs][specific_structure_index]["Point cloud"] = point_cloud
                    master_structure_reference_dict[patientUID][structs][specific_structure_index]["Delaunay triangulation"] = delaunay_triangulation_obj
                    
                    # test points to test for inclusion
                    num_pts = 5000
                    max_bnd = point_cloud.get_max_bound()
                    min_bnd = point_cloud.get_max_bound()
                    center = point_cloud.get_center()
                    if np.linalg.norm(max_bnd-center) >= np.linalg.norm(min_bnd-center): 
                        largest_bnd = max_bnd
                    else:
                        largest_bnd = min_bnd
                    bounding_box_size = np.linalg.norm(largest_bnd-center)
                    test_pts = [np.random.uniform(-bounding_box_size,bounding_box_size, size = 3) for i in range(num_pts)]
                    test_pts_arr = np.array(test_pts) + center
                    test_pts_point_cloud = o3d.geometry.PointCloud()
                    test_pts_point_cloud.points = o3d.utility.Vector3dVector(test_pts_arr)
                    test_pt_colors = np.empty([num_pts,3], dtype=float)

                    for ind,pts in enumerate(test_pts_arr):
                        #print(tri.find_simplex(pts) >= 0)  # True if point lies within poly)
                        if delaunay_triangulation_obj.delaunay_triangulation.find_simplex(pts) >= 0:
                            test_pt_colors[ind,:] = np.array([0,1,0]) # paint green
                        else: 
                            test_pt_colors[ind,:] = np.array([1,0,0]) # paint red
                    
                    test_pts_point_cloud.colors = o3d.utility.Vector3dVector(test_pt_colors)

                    # plot delaunay in open3d ?
                    #plotting_funcs.plot_tri_immediately_efficient(threeDdata_array, delaunay_triangulation_obj.delaunay_line_set, test_pts_point_cloud)
                    


                    if structs == structs_referenced_list[0]: 
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure centroid pts"] = structure_centroids_array


                        centroid_line = pca.linear_fitter(structure_centroids_array.T)
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Best fit line of centroid pts"] = centroid_line
                        
                        centroid_line_sample = np.array([centroid_line[0]])
                        num_centroid_samples_of_centroid_line = 20
                        travel_vec = np.array([centroid_line[1]-centroid_line[0]])*1/num_centroid_samples_of_centroid_line
                        for i in range(1,num_centroid_samples_of_centroid_line+1):
                            init_point = centroid_line_sample[-1]
                            new_point = init_point + travel_vec
                            centroid_line_sample=np.append(centroid_line_sample,new_point,axis=0)
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Centroid line sample pts"] = centroid_line_sample
  
                        # conduct a nearest neighbour search of biopsy centroids

                        #treescipy = scipy.spatial.KDTree(threeDdata_array)
                        #nn = treescipy.query(centroid_line_sample[0])
                        #nearest_neighbour = treescipy.data[nn[1]]

                        list_travel_vec = np.squeeze(travel_vec).tolist()
                        list_centroid_line_first_point = np.squeeze(centroid_line_sample[0]).tolist()
                        drawn_biopsy_array = biopsy_creator.biopsy_points_creater_by_transport(list_travel_vec,list_centroid_line_first_point,num_centroid_samples_of_centroid_line,np.linalg.norm(travel_vec),False)
                        master_structure_reference_dict[patientUID][structs][specific_structure_index]["Reconstructed structure pts"] = drawn_biopsy_array.T

                    # plot only the biopsies
                    if structs == structs_referenced_list[0]:
                        specific_structure["Plot attributes"].plot_bool = True
                    
                loader.iterator = loader.iterator + 1
            
    et = time.time()
    elapsed_time = et - st
    print('\n Execution time:', elapsed_time, 'seconds')


    # instantiate the variables used for the loading bar
    num_patients = len(master_structure_reference_dict)
    
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
                        #specific_structure_fig = plotting_funcs.arb_threeD_scatter_plotter(global_data_list_per_patient,**info)
                        specific_structure_fig = plotting_funcs.arb_threeD_scatter_plotter_list(global_data_list_per_patient,**info)
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
    

    # this is a user defined quantity, should be a tab delimited csv file in the future, mu sigma for each uncertainty direction
    # for now I will create some fake ones
    bx_num_uncertainties_x_bx_frame = 3 
    bx_num_uncertainties_y_bx_frame = 2 
    bx_num_uncertainties_z_bx_frame = 2 
    bx_uncertainties_means_x_bx_frame = np.full((bx_num_uncertainties_x_bx_frame,1), 0)
    bx_uncertainties_means_y_bx_frame = np.full((bx_num_uncertainties_y_bx_frame,1), 0)
    bx_uncertainties_means_z_bx_frame = np.full((bx_num_uncertainties_z_bx_frame,1), 0)

    bx_uncertainties_sigma_x_bx_frame = np.random.uniform(low=0.0, high=3.0, size=(bx_num_uncertainties_x_bx_frame,1))
    bx_uncertainties_sigma_y_bx_frame = np.random.uniform(low=0.0, high=3.0, size=(bx_num_uncertainties_y_bx_frame,1))
    bx_uncertainties_sigma_z_bx_frame = np.random.uniform(low=0.0, high=3.0, size=(bx_num_uncertainties_z_bx_frame,1))

    bx_uncertainties_mu_sigma_x_bx_frame = np.concatenate((bx_uncertainties_means_x_bx_frame, bx_uncertainties_sigma_x_bx_frame), axis=1)
    bx_uncertainties_mu_sigma_y_bx_frame = np.concatenate((bx_uncertainties_means_y_bx_frame, bx_uncertainties_sigma_y_bx_frame), axis=1)
    bx_uncertainties_mu_sigma_z_bx_frame = np.concatenate((bx_uncertainties_means_z_bx_frame, bx_uncertainties_sigma_z_bx_frame), axis=1)

    bx_uncertainties_mu_sigma_xy_bx_frame = np.concatenate((bx_uncertainties_mu_sigma_x_bx_frame, bx_uncertainties_mu_sigma_y_bx_frame), axis=1)
    bx_uncertainties_mu_sigma_xyz_bx_frame = np.concatenate((bx_uncertainties_mu_sigma_xy_bx_frame, bx_uncertainties_mu_sigma_z_bx_frame), axis=1)

    simulation_ans = ques_funcs.ask_ok('Everything is ready. Begin simulation?')
    
    if simulation_ans ==  True:
        print('Beginning simulation')
        master_structure_reference_dict_simulated = MC_simulator_convex.simulator(master_structure_reference_dict, structs_referenced_list)
    else: 
        pass

    print('Programme has ended.')

def UID_generator(pydicom_obj):
    UID_def = f"{str(pydicom_obj[0x0010,0x0010].value)} ({str(pydicom_obj[0x0010,0x0020].value)})"
    return UID_def


def structure_referencer(structure_dcm_dict, OAR_list,DIL_list,Bx_list):
    """
    A function that builds a reference library of the dicom elements passed to it so that 
    we can match the ROI name to the contour information, since the contour
    information is referenced to the name by a number.
    """
    master_st_ref_dict = {}
    ref_list = ["Bx ref","OAR ref","DIL ref"] # note that Bx ref has to be the first entry for other parts of the code to work!
    for UID, structure_item in structure_dcm_dict.items():
        bpsy_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Point cloud": None, "Delaunay triangulation": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "Random uniformly sampled volume pts": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
        OAR_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Point cloud": None, "Delaunay triangulation": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in OAR_list)]
        DIL_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Point cloud": None, "Delaunay triangulation": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in DIL_list)]
        master_st_ref_dict[UID] = {"Patient ID":str(structure_item[0x0010,0x0020].value),"Patient Name":str(structure_item[0x0010,0x0010].value),ref_list[0]:bpsy_ref, ref_list[1]:OAR_ref, ref_list[2]:DIL_ref,"Ready to plot data list": None}
    return master_st_ref_dict, ref_list


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


if __name__ == '__main__':    
    main()
    