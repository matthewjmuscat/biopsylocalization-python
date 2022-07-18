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

def main():
    """
    A programme designed to receive dicom data consisting of prostate 
    ultrasound containing contouring information. The programme is then 
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


                    # below code is deprecated
                    """
                    #st = time.time()
                    structure_centroids = [[],[],[]] # x values, y values, z values
                    for slice_object in RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:]:
                        contour_slice_points = slice_object.ContourData
                        #print(contour_slice_points)
                        twoDdata = []
                        twoDdata.append([float(y) for y in contour_slice_points[0::3]])
                        twoDdata.append([float(y) for y in contour_slice_points[1::3]])
                        zslice = contour_slice_points[2]
                        structure_slice_centroid = centroid_finder.centroid_finder_mean_based(twoDdata)
                        structure_centroids[0].append(structure_slice_centroid[0])
                        structure_centroids[1].append(structure_slice_centroid[1])
                        structure_centroids[2].append(zslice)

                    #et = time.time()
                    #elapsed_time = et - st
                    #print('\n Execution time:', elapsed_time, 'seconds')
                    structure_centroids_array_transpose = np.array(structure_centroids)
                    structure_centroids_array = structure_centroids_array_transpose.T
                    """

                    
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


                        # below code is deprecated
                        """
                        threeDdata = []
                        threeDdata.append([float(x) for y in RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:] for x in y.ContourData[0::3]])
                        threeDdata.append([float(x) for y in RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:] for x in y.ContourData[1::3]])
                        threeDdata.append([float(x) for y in RTst_dcms_dict[patientUID].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:] for x in y.ContourData[2::3]])
                        threeDdata_array = np.array(threeDdata).T
                        """

                        
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




    #RTst_dcms[0].StructureSetROISequence[0].ROIName
    #RTst_dcms[0].StructureSetROISequence[0].ROINumber
    #RTst_dcms[0].ROIContourSequence[0].ReferencedROINumber
    #RTst_dcms[0].ROIContourSequence[0].ContourSequence[0].ContourData

    #biopsy_localizer(RTst_dicom_item)
    
    global_data_list = []
    
    disp_figs = ques_funcs.ask_ok('Do you want to open all figures now?')
    
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
                    
                        specific_structure_fig.show()


                        
            global_data_list.append(global_data_list_per_patient)

        close_figs = ques_funcs.ask_to_continue("Press carriage return when you wish to close all figures")
        plt.close('all')
    else:
        pass
    


    figure_global = plotting_funcs.arb_threeD_scatter_plotter_global(global_data_list[0])
    figure_global.show()
    x=input()

    
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        info = {}
        info["Patient Name"] = pydicom_item["Patient Name"]
        info["Patient ID"] = pydicom_item["Patient ID"]
        figure_global_per_patient = plotting_funcs.plot_general_per_patient(pydicom_item, structs_referenced_list, OAR_plot_atr=[], **info)
        figure_global_per_patient.show()
        print('1')
        x=input()
    

    
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
        bpsy_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
        OAR_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in OAR_list)]
        DIL_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "KDtree": None, "Nearest neighbours objects": [], "Plot attributes": plot_attributes()} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in DIL_list)]
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


if __name__ == '__main__':    
    main()
    