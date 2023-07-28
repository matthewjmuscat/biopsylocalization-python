import pathlib
import csv
import loading_tools


def uncertainty_file_preper_global_sigma(uncertainties_file, master_structure_reference_dict, structs_referenced_list, num_general_structs, global_sigma):
    global_header = ['Total num structs']
    headerUID = ['Patient UID']
    headerSTRUCT = ['Structure type']
    headerROI = ['ROI']
    headerROIRefnum = ['Ref #']
    header_master_structure_reference_dict_specific_structure_index = ['Master ref dict specific structure index']
    headerFoR = ['Frame of reference']
    headerbx = 'Bx frame'
    headerLab = 'Lab frame'
    headerunc_bx = ['mu X (B)', 'sigma X (B)', 'mu Y (B)', 'sigma Y (B)', 'mu Z (B)', 'sigma Z (B)'] 
    headerunc_L = ['mu X (L)', 'sigma X (L)', 'mu Y (L)', 'sigma Y (L)', 'mu Z (L)', 'sigma Z (L)']
    headerunc_generic = ['mu X', 'sigma X', 'mu Y', 'sigma Y', 'mu Z', 'sigma Z']
    default_vals_row = [0,float(global_sigma),0,float(global_sigma),0,float(global_sigma)]
    end_struct = ['_']*6

   


    f = open(uncertainties_file, "w", newline='\n')
    writer = csv.writer(f)
    writer.writerow(global_header)
    global_header_data = [num_general_structs]
    writer.writerow(global_header_data)
    writer.writerow(end_struct)

    
    with loading_tools.Loader(num_general_structs,"Compiling uncertainty file...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            for structure_type in structs_referenced_list:
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                    headerrow = headerUID + headerSTRUCT + headerROI + headerROIRefnum + header_master_structure_reference_dict_specific_structure_index + headerFoR
                    if structure_type == structs_referenced_list[0]:
                        frame_of_reference = headerbx
                        #sub_header_row = headerunc_bx
                    else:
                        frame_of_reference = headerLab
                        #sub_header_row = headerunc_L
                    sub_header_row = headerunc_generic
                    header_data = [patientUID, structure_type, specific_structure["ROI"], specific_structure["Ref #"], specific_structure_index, frame_of_reference]
                    with open(uncertainties_file, "a", newline='\n') as f:
                        #writer = csv.writer(f)
                        writer.writerow(headerrow)
                        writer.writerow(header_data)
                        writer.writerow(sub_header_row)
                        writer.writerow(default_vals_row)
                        writer.writerow(end_struct)
                    loader.iterator = loader.iterator + 1




def uncertainty_file_preper_sigma_by_struct_type(uncertainties_file, master_structure_reference_dict, structs_referenced_list, num_general_structs, structs_referenced_dict):
    global_header = ['Total num structs']
    headerUID = ['Patient UID']
    headerSTRUCT = ['Structure type']
    headerROI = ['ROI']
    headerROIRefnum = ['Ref #']
    header_master_structure_reference_dict_specific_structure_index = ['Master ref dict specific structure index']
    headerFoR = ['Frame of reference']
    headerbx = 'Bx frame'
    headerLab = 'Lab frame'
    headerunc_bx = ['mu X (B)', 'sigma X (B)', 'mu Y (B)', 'sigma Y (B)', 'mu Z (B)', 'sigma Z (B)'] 
    headerunc_L = ['mu X (L)', 'sigma X (L)', 'mu Y (L)', 'sigma Y (L)', 'mu Z (L)', 'sigma Z (L)']
    headerunc_generic = ['mu X', 'sigma X', 'mu Y', 'sigma Y', 'mu Z', 'sigma Z']
    end_struct = ['_']*6

   


    f = open(uncertainties_file, "w", newline='\n')
    writer = csv.writer(f)
    writer.writerow(global_header)
    global_header_data = [num_general_structs]
    writer.writerow(global_header_data)
    writer.writerow(end_struct)

    
    with loading_tools.Loader(num_general_structs,"Compiling uncertainty file...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            for structure_type in structs_referenced_list:
                default_vals_row = [0, float(structs_referenced_dict[structure_type]["Default sigma"]), 0, float(structs_referenced_dict[structure_type]["Default sigma"]), 0, float(structs_referenced_dict[structure_type]["Default sigma"])]
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                    headerrow = headerUID + headerSTRUCT + headerROI + headerROIRefnum + header_master_structure_reference_dict_specific_structure_index + headerFoR
                    if structure_type == structs_referenced_list[0]:
                        frame_of_reference = headerbx
                        #sub_header_row = headerunc_bx
                    else:
                        frame_of_reference = headerLab
                        #sub_header_row = headerunc_L
                    sub_header_row = headerunc_generic
                    header_data = [patientUID, structure_type, specific_structure["ROI"], specific_structure["Ref #"], specific_structure_index, frame_of_reference]
                    with open(uncertainties_file, "a", newline='\n') as f:
                        #writer = csv.writer(f)
                        writer.writerow(headerrow)
                        writer.writerow(header_data)
                        writer.writerow(sub_header_row)
                        writer.writerow(default_vals_row)
                        writer.writerow(end_struct)
                    loader.iterator = loader.iterator + 1


 