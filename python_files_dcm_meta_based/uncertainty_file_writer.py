import pathlib
import csv
import loading_tools
import math
import misc_tools 
import pandas
import math_funcs
import numpy as np

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




def uncertainty_file_preper_sigma_by_struct_type(uncertainties_file, 
                                                 master_structure_reference_dict, 
                                                 structs_referenced_list, 
                                                 num_general_structs, 
                                                 structs_referenced_dict,
                                                 biopsy_variation_uncertainty_setting,
                                                 master_structure_info_dict):
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
                default_vals_row = [0, float(structs_referenced_dict[structure_type]["Default sigma X"]), 0, float(structs_referenced_dict[structure_type]["Default sigma Y"]), 0, float(structs_referenced_dict[structure_type]["Default sigma Z"])]
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
                    
                    # if include biopsy contour variation has been marked as included, then include this in the sigma value for the simulation
                    if biopsy_variation_uncertainty_setting == "Per biopsy max" and structure_type == structs_referenced_list[0]:
                        maximum_projected_variation_sp_biopsy = specific_structure['Maximum projected distance between original centroids']
                        sigma_x = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma X"])**2 + maximum_projected_variation_sp_biopsy**2)
                        sigma_y = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Y"])**2 + maximum_projected_variation_sp_biopsy**2)
                        sigma_z = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Z"])**2 + maximum_projected_variation_sp_biopsy**2)
                        
                    elif biopsy_variation_uncertainty_setting == "Global mean" and structure_type == structs_referenced_list[0]:
                        mean_variation_of_biopsy_centroids_cohort = master_structure_info_dict["Global"]["Mean biopsy centroid variation"]
                        sigma_x = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma X"])**2 + mean_variation_of_biopsy_centroids_cohort**2)
                        sigma_y = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Y"])**2 + mean_variation_of_biopsy_centroids_cohort**2)
                        sigma_z = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Z"])**2 + mean_variation_of_biopsy_centroids_cohort**2)

                    elif biopsy_variation_uncertainty_setting == "Default only" and structure_type == structs_referenced_list[0]:
                        mean_variation_of_biopsy_centroids_cohort = master_structure_info_dict["Global"]["Mean biopsy centroid variation"]
                        sigma_x = float(structs_referenced_dict[structure_type]["Default sigma X"])
                        sigma_y = float(structs_referenced_dict[structure_type]["Default sigma Y"])
                        sigma_z = float(structs_referenced_dict[structure_type]["Default sigma Z"])

                    default_vals_row = [0, sigma_x, 0, sigma_y, 0, sigma_z]
                    
                    with open(uncertainties_file, "a", newline='\n') as f:
                        #writer = csv.writer(f)
                        writer.writerow(headerrow)
                        writer.writerow(header_data)
                        writer.writerow(sub_header_row)
                        writer.writerow(default_vals_row)
                        writer.writerow(end_struct)
                    loader.iterator = loader.iterator + 1





def uncertainty_file_preper_by_struct_type_dataframe(master_structure_reference_dict, 
                                                 structs_referenced_list, 
                                                 structs_referenced_dict,
                                                 biopsy_variation_uncertainty_setting,
                                                 non_biopsy_variation_uncertainty_setting,
                                                 master_structure_info_dict):

    headerbx = 'Biopsy'
    headerLab = 'Lab'
    
    uncertainties_dataframe = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for structure_type in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                
                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure) 

                if structure_type == structs_referenced_list[0]:
                    frame_of_reference = headerbx
                    #sub_header_row = headerunc_bx
                else:
                    frame_of_reference = headerLab

                ### Biopsy handling
                if biopsy_variation_uncertainty_setting == "Per biopsy max" and structure_type == structs_referenced_list[0]:
                    maximum_projected_variation_sp_biopsy = specific_structure['Maximum projected distance between original centroids']
                    sigma_x = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma X"])**2 + maximum_projected_variation_sp_biopsy**2)
                    sigma_y = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Y"])**2 + maximum_projected_variation_sp_biopsy**2)
                    sigma_z = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Z"])**2 + maximum_projected_variation_sp_biopsy**2)
                    
                elif biopsy_variation_uncertainty_setting == "Global mean" and structure_type == structs_referenced_list[0]:
                    mean_variation_of_biopsy_centroids_cohort = master_structure_info_dict["Global"]["Mean biopsy centroid variation"]
                    sigma_x = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma X"])**2 + mean_variation_of_biopsy_centroids_cohort**2)
                    sigma_y = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Y"])**2 + mean_variation_of_biopsy_centroids_cohort**2)
                    sigma_z = math.sqrt(float(structs_referenced_dict[structure_type]["Default sigma Z"])**2 + mean_variation_of_biopsy_centroids_cohort**2)

                elif biopsy_variation_uncertainty_setting == "Default only" and structure_type == structs_referenced_list[0]:
                    mean_variation_of_biopsy_centroids_cohort = master_structure_info_dict["Global"]["Mean biopsy centroid variation"]
                    sigma_x = float(structs_referenced_dict[structure_type]["Default sigma X"])
                    sigma_y = float(structs_referenced_dict[structure_type]["Default sigma Y"])
                    sigma_z = float(structs_referenced_dict[structure_type]["Default sigma Z"])
                

                ### Not biopsy handling
                if non_biopsy_variation_uncertainty_setting == "Default only" and structure_type != structs_referenced_list[0]:
                    sigma_x = float(structs_referenced_dict[structure_type]["Default sigma X"])
                    sigma_y = float(structs_referenced_dict[structure_type]["Default sigma Y"])
                    sigma_z = float(structs_referenced_dict[structure_type]["Default sigma Z"])
                
                
                dict_for_dataframe = {"Patient UID": [patientUID],
                                      "Structure ID": [structure_info["Structure ID"]], 
                                      "Structure type": [structure_info["Struct ref type"]],
                                      "Structure dicom ref num": [structure_info["Dicom ref num"]], 
                                      "Structure index": [structure_info["Index number"]],
                                      "Frame of reference": [frame_of_reference],
                                      "mu (X)": [0], 
                                      "mu (Y)": [0], 
                                      "mu (Z)": [0], 
                                      "sigma (X)": [sigma_x], 
                                      "sigma (Y)": [sigma_y], 
                                      "sigma (Z)": [sigma_z] 
                                      }


                uncertainties_dataframe = pandas.concat([uncertainties_dataframe, pandas.DataFrame(dict_for_dataframe)])

    return uncertainties_dataframe



def uncertainty_file_preper_by_struct_type_dataframe_NEW(master_structure_reference_dict, 
                                                 structs_referenced_list, 
                                                 structs_referenced_dict,
                                                 biopsy_variation_uncertainty_setting,
                                                 non_biopsy_variation_uncertainty_setting,
                                                 use_added_in_quad_errors_as,
                                                 master_structure_info_dict):

    headerbx = 'Biopsy'
    headerLab = 'Lab'
    
    uncertainties_dataframe = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for structure_type in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structure_type]):
                
                structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_structure) 

                if structure_type == structs_referenced_list[0]:
                    frame_of_reference = headerbx
                    #sub_header_row = headerunc_bx
                else:
                    frame_of_reference = headerLab

                ### Biopsy handling
                if structure_type == structs_referenced_list[0]:
                    if biopsy_variation_uncertainty_setting == "Per biopsy max":
                        maximum_projected_variation_sp_biopsy = specific_structure['Maximum projected distance between original centroids']
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"] + [maximum_projected_variation_sp_biopsy])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"] + [maximum_projected_variation_sp_biopsy])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"] + [maximum_projected_variation_sp_biopsy])

                    elif biopsy_variation_uncertainty_setting == "Per biopsy mean":
                        mean_projected_variation_sp_biopsy = specific_structure['Mean centroid variation']
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"] + [mean_projected_variation_sp_biopsy])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"] + [mean_projected_variation_sp_biopsy])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"] + [mean_projected_variation_sp_biopsy])
                    
                    elif biopsy_variation_uncertainty_setting == "Global mean":
                        mean_variation_of_biopsy_centroids_cohort = master_structure_info_dict["Global"]["Mean biopsy centroid variation"]
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"] + [mean_variation_of_biopsy_centroids_cohort])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"] + [mean_variation_of_biopsy_centroids_cohort])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"] + [mean_variation_of_biopsy_centroids_cohort])

                    elif biopsy_variation_uncertainty_setting == "Default only":
                        mean_variation_of_biopsy_centroids_cohort = master_structure_info_dict["Global"]["Mean biopsy centroid variation"]
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"])
                    
                    else: #  Just default to default
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"])
                    

                ### Not biopsy handling
                elif structure_type != structs_referenced_list[0]:
                    if non_biopsy_variation_uncertainty_setting == "Default only":
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"])
                    
                    else: #  Just default to default
                        errs_X_arr = np.array(structs_referenced_dict[structure_type]["Default sigma X"])
                        errs_Y_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Y"])
                        errs_Z_arr = np.array(structs_referenced_dict[structure_type]["Default sigma Z"])
                
                if use_added_in_quad_errors_as == 'sigma':
                    sigma_x = math_funcs.add_in_quadrature(errs_X_arr)
                    sigma_y = math_funcs.add_in_quadrature(errs_Y_arr)
                    sigma_z = math_funcs.add_in_quadrature(errs_Z_arr)
                
                elif use_added_in_quad_errors_as == 'two sigma':
                    sigma_x = math_funcs.add_in_quadrature(errs_X_arr)/2
                    sigma_y = math_funcs.add_in_quadrature(errs_Y_arr)/2
                    sigma_z = math_funcs.add_in_quadrature(errs_Z_arr)/2
                

                dict_for_dataframe = {"Patient UID": [patientUID],
                                      "Structure ID": [structure_info["Structure ID"]], 
                                      "Structure type": [structure_info["Struct ref type"]],
                                      "Structure dicom ref num": [structure_info["Dicom ref num"]], 
                                      "Structure index": [structure_info["Index number"]],
                                      "Frame of reference": [frame_of_reference],
                                      "mu (X)": [0], 
                                      "mu (Y)": [0], 
                                      "mu (Z)": [0], 
                                      "sigma (X)": [sigma_x], 
                                      "sigma (Y)": [sigma_y], 
                                      "sigma (Z)": [sigma_z] 
                                      }


                uncertainties_dataframe = pandas.concat([uncertainties_dataframe, pandas.DataFrame(dict_for_dataframe)])

    return uncertainties_dataframe