# a programme designed to receive dicom data consisting of prostate ultrasound containing contouring information. The programme is then designed to analyse the contour information to localize the biopsy contours relative to the DIL and prostate contours

import numpy as np
import pydicom 
import pathlib

print('hello')
x= np.exp(1)
print(x)

data_folder = pathlib.Path("../../Data/patient 181/2022-03__Studies/DOE^JOHN_ANON181_RTst_2022-03-22_000000_._._n1__00000/")
patient_RTst_data = data_folder / "2.16.840.1.114362.1.6.5.5.15814.13653613585.608438520.320.99.dcm"
# ds = pydicom.filereader.dcmread(patient_RTst_data)
print('hello')



