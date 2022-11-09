import pathlib
import csv

Data_folder_name = 'Data'
Uncertainty_folder_name = 'Uncertainty data'
data_dir = pathlib.Path(__file__).parents[3].joinpath(Data_folder_name)
uncertainty_dir = data_dir.joinpath(Uncertainty_folder_name)
uncertainties_file = uncertainty_dir.joinpath("uncertainties_prepared_unfilled.csv")

headerUID = ['Patient UID']
headerROI = ['ROI']
headerbx = ['Bx frame']
headeranat = ['Lab frame']
headerunc_bx = ['mu X (B)', 'sigma X (B)', 'mu Y (B)', 'sigma Y (B)', 'mu Z (B)', 'sigma Z (B)'] 
headerunc_L = ['mu X (L)', 'sigma X (L)', 'mu Y (L)', 'sigma Y (L)', 'mu Z (L)', 'sigma Z (L)'] 


with open(uncertainties_file, "w", newline='\n') as f:
    headerrow = headerUID + headerROI + headerbx
    sub_header_row = headerunc_bx
    writer = csv.writer(f)
    writer.writerow(headerrow)
    writer.writerow(sub_header_row)


 