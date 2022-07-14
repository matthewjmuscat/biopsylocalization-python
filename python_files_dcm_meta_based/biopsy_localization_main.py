import pydicom # imported for reading dicom files
import pathlib # imported for navigating file system
import glob
import plotting_funcs
import matplotlib.pyplot as plt
import centroid_finder
import pca
import scipy
import numpy as np
import biopsy_creator
import sys # imported for loading bar
from decimal import Decimal # for use in the loading bar


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
    
    # the below is the creation of the PatientUID, that is generally created from or referenced from here throughout the programme, it is formed as "patientname (patientID)"
    RTst_dcms_dict = {str(x[0x0010,0x0010].value)+' ('+str(x[0x0010,0x0020].value)+')': x for x in dicom_elems_list if x[0x0008,0x0060].value == modality_list[0]}

    master_structure_reference_dict, structs_referenced_list = structure_referencer(RTst_dcms_dict, OAROI_contour_names,DIL_contour_names,Biopsy_contour_names)

    # Now, we dont want to add the contour points to the structure list above,
    # because the contour data is already stored in a data tree, which will allow
    # for faster processing when accessed and iterated. 
    

    # this dictionary determines which organs of which patient are to be plotted, in theory this could be user input
    fig_dict = {UID+specific_structure["ROI"]: True for UID, pydicom_item in master_structure_reference_dict.items() for structs in structs_referenced_list for specific_structure in pydicom_item[structs]}
    
    # build a data dictionary to store the data we extract and build about the patient
    data_dict = {UID: None for UID, pydicom_item in master_structure_reference_dict.items()}

    # instantiate the loading bar variable
    m_loading = 0 #for a loading bar
    num_patients = len(master_structure_reference_dict)

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        sys.stdout.write("%d%% complete..\r" % round(Decimal((m_loading/num_patients)*100),1)) # a loading bar

        #print(pydicom_item["Patient Name"]+' ('+pydicom_item["Patient ID"]+')')
        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                # The below print lines were just for my own understanding of how to access the data structure
                #print(specific_structure["ROI"])
                #print(specific_structure["Ref #"])
                #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0].ContourData)
                #print(RTst_dcms[dcm_index].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[1].ContourData)

                structure_centroids = [[],[],[]] # x values, y values, z values
                for slice_object in RTst_dcms_dict[pydicom_item['Patient Name']+' ('+pydicom_item['Patient ID']+')'].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:]:
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

                #print(structure_centroids)


                centroid_line = pca.linear_fitter(structure_centroids)

                centroid_line_sample = np.array([centroid_line[0]])
                # N is the number of sample points to be sampled along the centroid line
                num_centroid_samples_of_centroid_line = 20
                travel_vec = np.array([centroid_line[1]-centroid_line[0]])*1/num_centroid_samples_of_centroid_line
                for i in range(1,num_centroid_samples_of_centroid_line+1):
                    init_point = centroid_line_sample[-1]
                    new_point = init_point + travel_vec
                    centroid_line_sample=np.append(centroid_line_sample,new_point,axis=0)

                centroid_line_sample_transpose = centroid_line_sample.T
                centroid_line_sample_list = centroid_line_sample_transpose.tolist()
                centroid_line_sample_list_and_color = centroid_line_sample_list
                centroid_line_sample_list_and_color.append('y')
                centroid_line_sample_list_and_color.append('x')

                threeDdata = []
                threeDdata.append([float(x) for y in RTst_dcms_dict[pydicom_item['Patient Name']+' ('+pydicom_item['Patient ID']+')'].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:] for x in y.ContourData[0::3]])
                threeDdata.append([float(x) for y in RTst_dcms_dict[pydicom_item['Patient Name']+' ('+pydicom_item['Patient ID']+')'].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:] for x in y.ContourData[1::3]])
                threeDdata.append([float(x) for y in RTst_dcms_dict[pydicom_item['Patient Name']+' ('+pydicom_item['Patient ID']+')'].ROIContourSequence[int(specific_structure["Ref #"])].ContourSequence[0:] for x in y.ContourData[2::3]])
                threeDdata_array = np.array(threeDdata).T
                master_structure_reference_dict[patientUID][structs][specific_structure_index]["Raw contour pts"] = threeDdata_array

                #structure_centroids_array = np.array(structure_centroids).T
                #treescipy = scipy.spatial.KDTree(threeDdata_array)
                #nn = treescipy.query(structure_centroids_array[0])
                #nearest_neighbour = treescipy.data[nn[1]]
                

                list_travel_vec = np.squeeze(travel_vec).tolist()
                list_centroid_line_first_point = np.squeeze(centroid_line_sample[0]).tolist()
                drawn_biopsy_array = biopsy_creator.biopsy_points_creater_by_transport(list_travel_vec,list_centroid_line_first_point,num_centroid_samples_of_centroid_line,np.linalg.norm(travel_vec),False)
                drawn_biopsy_list = drawn_biopsy_array.tolist()
                drawn_biopsy_list_and_color = drawn_biopsy_list
                drawn_biopsy_list_and_color.append('m')
                drawn_biopsy_list_and_color.append('+')

                # only produces the plots that were specified as True by the fig_dict dictionary
                if fig_dict[pydicom_item["Patient Name"]+' ('+pydicom_item["Patient ID"]+') '+specific_structure["ROI"]] == True:
                    #specific_structure_fig = plotting_funcs.threeD_scatter_plotter(threeDdata[0],threeDdata[1],threeDdata[2])
                    threeDdata_and_color = threeDdata
                    threeDdata_and_color.append('r')
                    threeDdata_and_color.append('o')
                    structure_centroids_and_color = structure_centroids
                    structure_centroids_and_color.append('b')
                    structure_centroids_and_color.append('o')
                    info = specific_structure
                    #info.update()
                    info["Patient Name"] = pydicom_item["Patient Name"]
                    info["Patient ID"] = pydicom_item["Patient ID"]
                    specific_structure_fig = plotting_funcs.arb_threeD_scatter_plotter(threeDdata_and_color,structure_centroids_and_color,centroid_line_sample_list_and_color,drawn_biopsy_list_and_color,**info)
                    specific_structure_fig = plotting_funcs.add_line(specific_structure_fig,centroid_line)
                    fig_dict[pydicom_item["Patient Name"]+' ('+pydicom_item["Patient ID"]+') '+specific_structure["ROI"]] = specific_structure_fig
                    #fig_list.append([pydicom_item["Patient Name"]+' ('+pydicom_item["Patient ID"]+')'+specific_structure["ROI"], ])
        m_loading = m_loading + 1 # increasing loading bar iterator
        sys.stdout.flush() #flushing the sys output (rewriting over previous loading bar)
    print("100% complete.")




    #RTst_dcms[0].StructureSetROISequence[0].ROIName
    #RTst_dcms[0].StructureSetROISequence[0].ROINumber
    #RTst_dcms[0].ROIContourSequence[0].ReferencedROINumber
    #RTst_dcms[0].ROIContourSequence[0].ContourSequence[0].ContourData

    #biopsy_localizer(RTst_dicom_item)
    

    
    
    
    
    for key, value in fig_dict.items():
        value.show()
    
    x= input()
    #def biopsy_localizer():


def structure_referencer(structure_dcm_dict, OAR_list,DIL_list,Bx_list):
    """
    A function that builds a reference library of the dicom elements passed to it so that we can match the ROI name to the contour information, since the contour
    information is referenced to the name by a number.
    """
    master_st_ref_dict = {}
    ref_list = ["Bx ref","OAR ref","DIL ref"]
    for UID, structure_item in structure_dcm_dict.items():
        bpsy_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "Nearest neighbour pts": None} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in Bx_list)]    
        OAR_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "Nearest neighbour pts": None} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in OAR_list)]
        DIL_ref = [{"ROI":x.ROIName, "Ref #":x.ROINumber, "Raw contour pts": None, "Structure centroid pts": None, "Best fit line of centroid pts": None, "Centroid line sample pts": None, "Reconstructed structure pts": None, "Nearest neighbour pts": None} for x in structure_item.StructureSetROISequence if any(i in x.ROIName for i in DIL_list)]
        master_st_ref_dict[UID] = {"Patient ID":str(structure_item[0x0010,0x0020].value),"Patient Name":str(structure_item[0x0010,0x0010].value),ref_list[0]:bpsy_ref, ref_list[1]:OAR_ref, ref_list[2]:DIL_ref}
    return master_st_ref_dict, ref_list









if __name__ == '__main__':    
    main()
    