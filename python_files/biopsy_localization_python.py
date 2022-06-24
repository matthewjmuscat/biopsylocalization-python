# A programme designed to receive dicom data consisting of prostate 
# ultrasound containing contouring information. The programme is then 
# designed to analyse the contour information to localize the biopsy 
# contours relative to the DIL and prostate contours

import numpy as np # imported for math and arrays
import pydicom # imported for reading dicom files
#import pathlib # imported for navigating file system
#import sys # imported for loading bar
#from decimal import Decimal # for use in the loading bar
#import time # allows function to tell programme to wait, this was for testing the loading bar 
import ques_funcs # the libary I made containing question functions directed towards the user
import data_library_builder


def main():
    """
    A programme designed to receive dicom data consisting of prostate 
    ultrasound containing contouring information. The programme is then 
    designed to analyse the contour information to localize the biopsy 
    contours relative to the DIL and prostate contours
    """
    # checkpoint 0
    print('checkpoint 0')

    # create the lookup table for the contour structure folder names within the Data folder
    contour_structure_lookup_table_input = ['RTst']
    

    # checkpoint 1
    print('checkpoint 1')

    # ask to skip file system check
    skip_fsc = ques_funcs.ask_ok('Do you want to skip the file system check?')

    if skip_fsc == True:
        print(skip_fsc)
    elif skip_fsc == False:
        print(skip_fsc)
    else:
        print('does this execute?')
    
    # run the data library builder
    patient_data = data_library_builder.data_lib_bldr(contour_structure_lookup_table_input)
    
    #,'patient 181','2022-03__Studies','DOE^JOHN_ANON181_RTst_2022-03-22_000000_._._n1__00000')
    #Path_patient_RTst_dcm = Path_data_folder.joinpath("2.16.840.1.114362.1.6.5.5.15814.13653613585.608438520.320.99.dcm")
    #RTst_dcm = pydicom.filereader.dcmread(Path_patient_RTst_dcm)





    # checkpoint 2
    print('checkpoint 2')


if __name__ == '__main__':    
    main()
    x = 4


