# a programme designed to receive dicom data consisting of prostate 
# ultrasound containing contouring information. The programme is then 
# designed to analyse the contour information to localize the biopsy 
# contours relative to the DIL and prostate contours

import numpy as np # imported for math and arrays
import pydicom # imported for reading dicom files
import pathlib # imported for navigating file system
import sys # imported for loading bar
from decimal import Decimal # for use in the loading bar
import time # allows function to tell programme to wait, this was for testing the loading bar 
import ques_funcs # the libary I made containing question functions directed towards the user

# create the lookup table for the contour structure folder names within the Data folder
contour_structure_lookup_table = ['RTst']


def main():

    # checkpoint 0
    print('checkpoint 0')

    # instantiate the loading bar variable
    m_loading = 0 #for a loading bar

    # checkpoint 1
    print('checkpoint 1')






    # need to ensure that the patient data folder is up one level from the project folder, AND that the python files are down one level from the project folder (ie. in python_files)
    # set the variables guiding the path to the data
    global_patient_list = [['patient_id','patient_path','treatment_id','treatment_path','RTst_path','RTDOSE_path']] # build the first entry of global patient path list showing what each entry corresponds to 
    Path_two_levels_up = pathlib.Path(__file__).parents[2] # the path two levels up from the location of this python file
    Path_data_folder = Path_two_levels_up.joinpath('Data') # join the Data folder to the two levels up path
    Path_list_patients = [x for x in Path_data_folder.iterdir() if x.is_dir()] # generate a list containing the paths of each patient stored in the data folder

    # construct a data structure containing the patient id, treatment id, contour data path, dose data path
    num_patients = len(Path_list_patients) # for loading bar
    print("Building data library..")
    for patient in enumerate(Path_list_patients): # loop over each patient, enumerate also returns the iteration number 
        sys.stdout.write("%d%% complete..\r" % round(Decimal((m_loading/num_patients)*100),1)) # a loading bar
        sub_patient_list = [[Path_list_patients[patient[0]].name]] # create a local list for each iteration to be appended to the global list, create the first entry in the list to be the patients name, which is actually the directory name for that patient
        sub_patient_list[0].append(patient[1])
        Path_list_treatments = [x for x in patient[1].iterdir() if x.is_dir()] # create a local list containing the paths of each treatment within a patient
        sub_patient_list[0].append(Path_list_treatments[0].name)  # the first entry in the list of treatments should be the first treatment for that patient
        if len(Path_list_treatments) > 1: # if the list of treatments is greater than 1 for a patient, then loop over treatments except the first
            for treatment in enumerate(Path_list_treatments[1:],1):
                sub_patient_list.append([sub_patient_list[0][0],sub_patient_list[0][1]]) # create the same name and path of the patient for the new list containing the other treatment
                sub_patient_list[treatment[0]].append(Path_list_treatments[treatment[0]].name) # for each iteration append the treatment number for that sublist
        for treatment_entry in enumerate(sub_patient_list): # loop over all subtreatments for that particular patient
            sub_patient_list[treatment_entry[0]].append(sub_patient_list[treatment_entry[0]][1].joinpath(sub_patient_list[treatment_entry[0]][2])) # for the structure path, include the treatment to the file path
            RTst_data_in_treatment = [x for x in sub_patient_list[treatment_entry[0]][3].iterdir() if any(i in str(x) for i in contour_structure_lookup_table)] # generate a list containing the paths of each patient stored in the data folder
            sub_patient_list[treatment_entry[0]].append(RTst_data_in_treatment)
        for treatment_entry in enumerate(sub_patient_list):
            global_patient_list.append(treatment_entry[1]) # append each subtreatment list to the global list
        time.sleep(5)
        m_loading = m_loading + 1 # increasing loading bar iterator
        sys.stdout.flush() #flushing the sys output (rewriting over previous loading bar)
    print("100% complete.")
    #,'patient 181','2022-03__Studies','DOE^JOHN_ANON181_RTst_2022-03-22_000000_._._n1__00000')
    #Path_patient_RTst_dcm = Path_data_folder.joinpath("2.16.840.1.114362.1.6.5.5.15814.13653613585.608438520.320.99.dcm")
    #RTst_dcm = pydicom.filereader.dcmread(Path_patient_RTst_dcm)





    # checkpoint 2
    print('checkpoint 2')


if __name__ == '__main__':    main()



