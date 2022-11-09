import pathlib
import csv
import loading_tools

def uncertainty_file_preper(uncertainties_file, master_structure_reference_dict, structs_referenced_list, num_general_structs):

    headerUID = ['Patient UID']
    headerROI = ['ROI']
    headerROIRefnum = ['Ref #']
    headerFoR = ['Frame of reference']
    headerbx = 'Bx frame'
    headerLab = 'Lab frame'
    headerunc_bx = ['mu X (B)', 'sigma X (B)', 'mu Y (B)', 'sigma Y (B)', 'mu Z (B)', 'sigma Z (B)'] 
    headerunc_L = ['mu X (L)', 'sigma X (L)', 'mu Y (L)', 'sigma Y (L)', 'mu Z (L)', 'sigma Z (L)']
    zero_row = [0]*6
    end_struct = ['_']*6

    f = open(uncertainties_file, "w", newline='\n')

    with loading_tools.Loader(num_general_structs,"Compiling uncertainty file...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            for structs in structs_referenced_list:
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                    headerrow = headerUID + headerROI + headerROIRefnum + headerFoR
                    if structs == structs_referenced_list[0]:
                        frame_of_reference = headerbx
                        sub_header_row = headerunc_bx
                    else:
                        frame_of_reference = headerLab
                        sub_header_row = headerunc_L
                    
                    header_data = [patientUID, specific_structure["ROI"], specific_structure["Ref #"], frame_of_reference]
                    with open(uncertainties_file, "a", newline='\n') as f:
                        writer = csv.writer(f)
                        writer.writerow(headerrow)
                        writer.writerow(header_data)
                        writer.writerow(sub_header_row)
                        writer.writerow(zero_row)
                        writer.writerow(end_struct)


 