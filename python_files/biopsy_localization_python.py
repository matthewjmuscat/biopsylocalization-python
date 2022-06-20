# a programme designed to receive dicom data consisting of prostate ultrasound containing contouring information. The programme is then designed to analyse the contour information to localize the biopsy contours relative to the DIL and prostate contours

import numpy as np
import pydicom 
import pathlib

# checkpoint 1
print('checkpoint 1')
x= np.exp(1)
print(x)

# need to ensure that the patient data folder is up one level from the project folder, AND that the python files are down one level from the project folder (ie. in python_files)
Path_two_levels_up = pathlib.Path(__file__).parents[2]
Path_data_folder = Path_two_levels_up.joinpath('Data','patient 181','2022-03__Studies','DOE^JOHN_ANON181_RTst_2022-03-22_000000_._._n1__00000')
Path_patient_RTst_dcm = Path_data_folder.joinpath("2.16.840.1.114362.1.6.5.5.15814.13653613585.608438520.320.99.dcm")
RTst_dcm = pydicom.filereader.dcmread(Path_patient_RTst_dcm)

# checkpoint 2
print('checkpoint 2')



