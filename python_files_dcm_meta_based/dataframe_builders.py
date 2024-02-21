import numpy as np
import pandas 
import csv

def tissue_probability_dataframe_builder_by_bx_pt(specific_bx_structure, 
                                         structure_miss_probability_roi,
                                         cancer_tissue_label,
                                         miss_structure_complement_label
                                         ):
    
    bx_struct_roi = specific_bx_structure["ROI"]
    bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
    bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
    bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
    pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

    tumor_tissue_bionomial_est_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue binomial est arr"]
    tumor_tissue_bionomial_se_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue standard error arr"]
    tumor_tissue_conf_int_2d_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue confidence interval 95 arr"]
    tumor_tissue_nominal_containment_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue nominal arr"]
    tumor_tissue_conf_int_lower_arr = tumor_tissue_conf_int_2d_arr[0,:]
    tumor_tissue_conf_int_upper_arr = tumor_tissue_conf_int_2d_arr[1,:]

    pt_radius_point_wise_for_pd_data_frame_list = pt_radius_bx_coord_sys.tolist()
    axial_Z_point_wise_for_pd_data_frame_list = bx_points_bx_coords_sys_arr[:,2].tolist()
    binom_est_point_wise_for_pd_data_frame_list = tumor_tissue_bionomial_est_arr.tolist()
    std_err_point_wise_for_pd_data_frame_list = tumor_tissue_bionomial_se_arr.tolist()
    ROI_name_point_wise_for_pd_data_frame_list = [cancer_tissue_label]*len(bx_points_bx_coords_sys_arr_list)
    nominal_point_wise_for_pd_data_frame_list = tumor_tissue_nominal_containment_arr.tolist()
    binom_est_lower_CI_point_wise_for_pd_data_frame_list = tumor_tissue_conf_int_lower_arr.tolist()
    binom_est_upper_CI_point_wise_for_pd_data_frame_list = tumor_tissue_conf_int_upper_arr.tolist()
            
    for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
        containment_structure_ROI = containment_structure_key_tuple[0]
        if structure_miss_probability_roi not in containment_structure_ROI:
            continue
        
        ROI_name_point_wise_for_pd_data_frame_list = ROI_name_point_wise_for_pd_data_frame_list + [containment_structure_ROI]*len(bx_points_bx_coords_sys_arr_list)
        containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
        containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
        containment_structure_CI_list_of_tuples = containment_structure_dict["Confidence interval 95 (containment) list"]
        conf_int_lower_list = [upper_lower_tup[0] for upper_lower_tup in containment_structure_CI_list_of_tuples]
        conf_int_upper_list = [upper_lower_tup[1] for upper_lower_tup in containment_structure_CI_list_of_tuples]
        containment_structure_nominal_list = containment_structure_dict["Nominal containment list"]
        binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + containment_structure_binom_est_list
        std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + containment_structure_stand_err_list
        binom_est_lower_CI_point_wise_for_pd_data_frame_list = binom_est_lower_CI_point_wise_for_pd_data_frame_list + conf_int_lower_list
        binom_est_upper_CI_point_wise_for_pd_data_frame_list = binom_est_upper_CI_point_wise_for_pd_data_frame_list + conf_int_upper_list
        nominal_point_wise_for_pd_data_frame_list = nominal_point_wise_for_pd_data_frame_list + containment_structure_nominal_list
        
        pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + pt_radius_bx_coord_sys.tolist()
        axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,2].tolist() 

    # include complemenet of miss structure        
    specific_bx_structure_relative_OAR_dict = specific_bx_structure["MC data: miss structure tissue probability"]
    miss_structure_binom_est_list = specific_bx_structure_relative_OAR_dict["OAR tissue miss binomial est arr"].tolist()
    miss_structure_standard_err_list = specific_bx_structure_relative_OAR_dict["OAR tissue standard error arr"].tolist()
    miss_structure_CI_2d_arr = specific_bx_structure_relative_OAR_dict["OAR tissue miss confidence interval 95 2d arr"]
    miss_structure_CI_lower_list = miss_structure_CI_2d_arr[0,:].tolist()
    miss_structure_CI_upper_list = miss_structure_CI_2d_arr[1,:].tolist()
    miss_structure_nominal_list = specific_bx_structure_relative_OAR_dict["OAR tissue miss nominal arr"].tolist()

    ROI_name_point_wise_for_pd_data_frame_list = ROI_name_point_wise_for_pd_data_frame_list + [miss_structure_complement_label]*len(bx_points_bx_coords_sys_arr_list)
    pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + pt_radius_bx_coord_sys.tolist()
    axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,2].tolist() 
    binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + miss_structure_binom_est_list
    std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + miss_structure_standard_err_list
    binom_est_lower_CI_point_wise_for_pd_data_frame_list = binom_est_lower_CI_point_wise_for_pd_data_frame_list + miss_structure_CI_lower_list
    binom_est_upper_CI_point_wise_for_pd_data_frame_list = binom_est_upper_CI_point_wise_for_pd_data_frame_list + miss_structure_CI_upper_list
    nominal_point_wise_for_pd_data_frame_list = nominal_point_wise_for_pd_data_frame_list + miss_structure_nominal_list
        
    containment_output_dict_by_MC_trial_for_pandas_data_frame = {"Bx structure ROI": bx_struct_roi,
                                                                "Structure ROI": ROI_name_point_wise_for_pd_data_frame_list, 
                                                                "Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, 
                                                                "STD err": std_err_point_wise_for_pd_data_frame_list,
                                                                "Nominal containment": nominal_point_wise_for_pd_data_frame_list,
                                                                "CI lower vals": binom_est_lower_CI_point_wise_for_pd_data_frame_list,
                                                                "CI upper vals": binom_est_upper_CI_point_wise_for_pd_data_frame_list
                                                                }

    containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)

    return containment_output_dict_by_MC_trial_for_pandas_data_frame, containment_output_by_MC_trial_pandas_data_frame




def containment_global_scores_all_patients_dataframe_builder(all_patient_sub_dirs):
    data_frame_list = []
    num_actual_biopsies = 0
    num_sim_biopsies = 0
    for directory in all_patient_sub_dirs:
        csv_files_in_directory_list = list(directory.glob('*.csv'))
        containment_csvs_list = [csv_file for csv_file in csv_files_in_directory_list if "containment_out" in csv_file.name]
        for contianment_csv in containment_csvs_list:
            with open(contianment_csv, "r", newline='\n') as contianment_csv_open:
                sample_dict = {}
                reader_obj_list = list(csv.reader(contianment_csv_open))
                info = reader_obj_list[0:3]
                patient_id = info[0][1]
                bx_id = info[1][1]
                simulated_string = info[2][1]
                if simulated_string.lower() == 'false':
                    simulated_bool = False 
                    num_actual_biopsies = num_actual_biopsies + 1
                else: 
                    simulated_bool = True
                    num_sim_biopsies = num_sim_biopsies + 1

                for row_index,row in enumerate(reader_obj_list):
                    if "Global by class" in row:
                        starting_index = row_index + 2
                        break
                    pass
                
                tissue_iteration = 0
                sample_dict["Patient ID"] = []
                sample_dict["Bx ID"] = []
                sample_dict["Simulated bool"] = []
                for row_index, row in enumerate(reader_obj_list[starting_index:]):
                    if "+++" in row:
                        tissue_iteration = tissue_iteration + 1
                        sample_dict["Patient ID"].append(patient_id)
                        sample_dict["Bx ID"].append(bx_id)
                        sample_dict["Simulated bool"].append(simulated_bool)
                        continue
                    if "---" in row:
                        break
                    
                    if tissue_iteration == 1:
                        if row[0] == 'Tissue type':
                            sample_dict[row[0]] = [row[1]]
                        else:
                            sample_dict[row[0]] = [float(row[1])]
                        
                    else:
                        if row[0] == 'Tissue type':
                            sample_dict[row[0]].append(row[1])
                        else:
                            sample_dict[row[0]].append(float(row[1]))

                bx_sp_dataframe = pandas.DataFrame(data=sample_dict)
                data_frame_list.append(bx_sp_dataframe)

    cohort_containment_dataframe = pandas.concat(data_frame_list,ignore_index = True)   

    return num_actual_biopsies, num_sim_biopsies, cohort_containment_dataframe


def tissue_length_by_threshold_all_patients_dataframe_builder(all_patient_sub_dirs):
    data_frame_list = []
    num_actual_biopsies = 0
    num_sim_biopsies = 0
    for directory in all_patient_sub_dirs:
        csv_files_in_directory_list = list(directory.glob('*.csv'))
        containment_csvs_list = [csv_file for csv_file in csv_files_in_directory_list if "containment_out" in csv_file.name]
        for contianment_csv in containment_csvs_list:
            with open(contianment_csv, "r", newline='\n') as contianment_csv_open:
                sample_dict = {}
                reader_obj_list = list(csv.reader(contianment_csv_open))
                info = reader_obj_list[0:3]
                patient_id = info[0][1]
                bx_id = info[1][1]
                simulated_string = info[2][1]
                if simulated_string.lower() == 'false':
                    simulated_bool = False 
                    num_actual_biopsies = num_actual_biopsies + 1
                else: 
                    simulated_bool = True
                    num_sim_biopsies = num_sim_biopsies + 1

                for row_index,row in enumerate(reader_obj_list):
                    if "Tumor length estimate" in row:
                        starting_index = row_index + 2
                        break
                    pass
                
                threshold_iteration = 0
                sample_dict["Patient ID"] = []
                sample_dict["Bx ID"] = []
                sample_dict["Simulated bool"] = []
                for row_index, row in enumerate(reader_obj_list[starting_index:]):
                    if "+++" in row:
                        threshold_iteration = threshold_iteration + 1
                        sample_dict["Patient ID"].append(patient_id)
                        sample_dict["Bx ID"].append(bx_id)
                        sample_dict["Simulated bool"].append(simulated_bool)
                        continue
                    if "---" in row:
                        break
                    
                    if threshold_iteration == 1:
                        if "Length estimate bootstrap distribution" in row:
                            pass
                        else:
                            sample_dict[row[0]] = [float(row[1])]     
                    else:
                        if "Length estimate bootstrap distribution" in row:
                            pass
                        else:
                            sample_dict[row[0]].append(float(row[1]))

                bx_sp_dataframe = pandas.DataFrame(data=sample_dict)
                data_frame_list.append(bx_sp_dataframe)

    cohort_containment_dataframe = pandas.concat(data_frame_list,ignore_index = True)   

    return num_actual_biopsies, num_sim_biopsies, cohort_containment_dataframe


def cumulative_histogram_for_tissue_length_dataframe_builder(patient_cohort_dataframe,
                                                             threshold):
    
    x_sim = patient_cohort_dataframe[(patient_cohort_dataframe["Simulated bool"] == True) & (patient_cohort_dataframe["Probability threshold"] == threshold)]["Length estimate mean"].to_numpy()
    x_actual = patient_cohort_dataframe[(patient_cohort_dataframe["Simulated bool"] == False) & (patient_cohort_dataframe["Probability threshold"] == threshold)]["Length estimate mean"].to_numpy()
    
    
    hist_sim, bin_edges_sim = np.histogram(x_sim, bins=100, density=True)
    cdf_sim = np.cumsum(hist_sim * np.diff(bin_edges_sim))

    hist_actual, bin_edges_actual = np.histogram(x_actual, bins=100, density=True)
    cdf_actual = np.cumsum(hist_actual * np.diff(bin_edges_actual))

    cdf_dict = {"CDF sim": {"Bin edges": bin_edges_sim,
                            "CDF": cdf_sim},
                "CDF actual": {"Bin edges": bin_edges_actual,
                               "CDF": cdf_actual}
    }
    
    return cdf_dict
    



def dose_global_scores_all_patients_dataframe_builder(all_patient_sub_dirs):
    data_frame_list = []
    num_actual_biopsies = 0
    num_sim_biopsies = 0
    for directory in all_patient_sub_dirs:
        csv_files_in_directory_list = list(directory.glob('*.csv'))
        dose_csvs_list = [csv_file for csv_file in csv_files_in_directory_list if "dose_out" in csv_file.name]
        for dose_csv in dose_csvs_list:
            with open(dose_csv, "r", newline='\n') as dose_csv_open:
                sample_dict = {}
                reader_obj_list = list(csv.reader(dose_csv_open))
                info = reader_obj_list[0:3]
                patient_id = info[0][1]
                bx_id = info[1][1]
                simulated_string = info[2][1]
                if simulated_string.lower() == 'false':
                    simulated_bool = False 
                    num_actual_biopsies = num_actual_biopsies + 1
                else: 
                    simulated_bool = True
                    num_sim_biopsies = num_sim_biopsies + 1

                for row_index,row in enumerate(reader_obj_list):
                    if "Global" in row:
                        starting_index = row_index + 2
                        break
                    pass
                
                #tissue_iteration = 1
                sample_dict["Patient ID"] = [patient_id]
                sample_dict["Bx ID"] = [bx_id]
                sample_dict["Simulated bool"] = [simulated_bool]
                
                for row_index, row in enumerate(reader_obj_list[starting_index:]):
                    if row[0] == '---':
                        break
                    sample_dict[row[0]] = [float(row[1])]

                bx_sp_dataframe = pandas.DataFrame(data=sample_dict)
                data_frame_list.append(bx_sp_dataframe)

    cohort_dose_dataframe = pandas.concat(data_frame_list,ignore_index = True)   

    return num_actual_biopsies, num_sim_biopsies, cohort_dose_dataframe



def all_dose_data_by_trial_and_pt_from_MC_trial_dataframe_builder(master_structure_ref_dict,
                                                                  bx_ref
                                                                  ):
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            dose_output_z_and_radius_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
            pt_radius_bx_coord_sys = dose_output_z_and_radius_dict_for_pandas_data_frame["Radial pos (mm)"]

            bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
            #bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
            #pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)
            


            # create a 2d scatter plot with all MC trials on plot
            dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"]
            dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_list = dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr.tolist()
            pt_radius_point_wise_for_pd_data_frame_list = []
            axial_Z_point_wise_for_pd_data_frame_list = []
            dose_vals_point_wise_for_pd_data_frame_list = []
            MC_trial_index_point_wise_for_pd_data_frame_list = []
            num_nominal_and_all_MC_trials = dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr.shape[1]
            for pt_index, specific_pt_all_MC_dose_vals in enumerate(dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_list):
                pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + [pt_radius_bx_coord_sys[pt_index]]*num_nominal_and_all_MC_trials
                axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + [bx_points_bx_coords_sys_arr[pt_index,2]]*num_nominal_and_all_MC_trials
                dose_vals_point_wise_for_pd_data_frame_list = dose_vals_point_wise_for_pd_data_frame_list + specific_pt_all_MC_dose_vals
                MC_trial_index_point_wise_for_pd_data_frame_list = MC_trial_index_point_wise_for_pd_data_frame_list + list(range(0,num_nominal_and_all_MC_trials))
            
            # Note that the 0th MC trial num index is the nominal value
            dose_output_dict_by_MC_trial_for_pandas_data_frame = {"Radial pos (mm)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                "Axial pos Z (mm)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                "Dose (Gy)": dose_vals_point_wise_for_pd_data_frame_list, 
                                                                "MC trial num": MC_trial_index_point_wise_for_pd_data_frame_list
                                                                }
            
            dose_output_nominal_and_all_MC_trials_pandas_data_frame = pandas.DataFrame.from_dict(data = dose_output_dict_by_MC_trial_for_pandas_data_frame)
            specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"] = dose_output_nominal_and_all_MC_trials_pandas_data_frame
            specific_bx_structure["Output dicts for data frames"]["Point-wise dose output by MC trial number"] = dose_output_dict_by_MC_trial_for_pandas_data_frame




def structure_volume_dataframe_builder(master_structure_ref_dict,
                                       structs_referenced_list,
                                       all_ref_key):
    

    for patientUID,pydicom_item in master_structure_ref_dict.items():
        structure_ID_list = []
        structure_type_list = []
        structure_volume_list = []
        structure_max_pairwise_dist_list = []
        structure_ref_num_list = []
        structure_index_list = []
        patient_ID_list = []
        voxel_size_list = []
        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                structureID = specific_structure["ROI"]
                structure_reference_number = specific_structure["Ref #"]
                struct_type = structs

                maximum_distance = specific_structure["Maximum pairwise distance"]
                structure_volume = specific_structure["Structure volume"]
                voxel_size = specific_structure["Voxel size for structure volume calc"]

                structure_ID_list.append(structureID)
                structure_type_list.append(struct_type)
                structure_ref_num_list.append(structure_reference_number)
                structure_max_pairwise_dist_list.append(maximum_distance)
                structure_volume_list.append(structure_volume)
                structure_index_list.append(specific_structure_index)
                patient_ID_list.append(patientUID)
                voxel_size_list.append(voxel_size)

        structure_info_dict_for_pandas_dataframe = {"Patient UID": patient_ID_list,
                                                    "Structure ID": structure_ID_list,
                                                    "Structure type": structure_type_list,
                                                    "Structure ref num": structure_ref_num_list,
                                                    "Structure index": structure_index_list,
                                                    "Structure max pair-wise distance": structure_max_pairwise_dist_list,
                                                    "Structure volume": structure_volume_list,
                                                    "Voxel side length": voxel_size_list
                                                    }
        
        structure_info_pandas_data_frame = pandas.DataFrame.from_dict(data = structure_info_dict_for_pandas_dataframe)

        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Structure information"] = structure_info_pandas_data_frame



def structure_dimension_dataframe_builder(master_structure_ref_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref):
    

    for patientUID,pydicom_item in master_structure_ref_dict.items():
        structure_ID_list = []
        structure_type_list = []
        structure_x_dim_list = []
        structure_y_dim_list = []
        structure_z_dim_list = []
        structure_ref_num_list = []
        structure_index_list = []
        patient_ID_list = []
        voxel_size_list = []
        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                if structs != bx_ref:
                    structureID = specific_structure["ROI"]
                    structure_reference_number = specific_structure["Ref #"]
                    struct_type = structs

                    structure_dimension_dict = specific_structure["Structure dimension at centroid dict"]
                    x_dim_len = structure_dimension_dict["X dimension length at centroid"]
                    y_dim_len = structure_dimension_dict["Y dimension length at centroid"]
                    z_dim_len = structure_dimension_dict["Z dimension length at centroid"]

                    voxel_size = specific_structure["Voxel size for structure dimension calc"]

                    structure_ID_list.append(structureID)
                    structure_type_list.append(struct_type)
                    structure_ref_num_list.append(structure_reference_number)
                    structure_x_dim_list.append(x_dim_len)
                    structure_y_dim_list.append(y_dim_len)
                    structure_z_dim_list.append(z_dim_len)
                    structure_index_list.append(specific_structure_index)
                    patient_ID_list.append(patientUID)
                    voxel_size_list.append(voxel_size)
                else: 
                    pass

        structure_info_dict_for_pandas_dataframe = {"Patient UID": patient_ID_list,
                                                    "Structure ID": structure_ID_list,
                                                    "Structure type": structure_type_list,
                                                    "Structure ref num": structure_ref_num_list,
                                                    "Structure index": structure_index_list,
                                                    "Structure X dim length (mm)": structure_x_dim_list,
                                                    "Structure Y dim length (mm)": structure_y_dim_list,
                                                    "Structure Z dim length (mm)": structure_z_dim_list,
                                                    "Voxel side length": voxel_size_list
                                                    }
        
        structure_info_pandas_data_frame = pandas.DataFrame.from_dict(data = structure_info_dict_for_pandas_dataframe)

        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Structure information dimension"] = structure_info_pandas_data_frame



def non_bx_structure_info_dataframe_builder(master_structure_ref_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref):
    

    for patientUID,pydicom_item in master_structure_ref_dict.items():
        patient_ID_list = []
        structure_ID_list = []
        structure_type_list = []
        structure_ref_num_list = []
        structure_index_list = []

        structure_volume_list = []
        structure_max_pairwise_dist_list = []
        voxel_size_for_volume_list = []

        structure_x_dim_list = []
        structure_y_dim_list = []
        structure_z_dim_list = []
        voxel_size_for_dimension_list = []

        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                if structs != bx_ref:
                    structureID = specific_structure["ROI"]
                    structure_reference_number = specific_structure["Ref #"]
                    struct_type = structs

                    maximum_distance = specific_structure["Maximum pairwise distance"]
                    structure_volume = specific_structure["Structure volume"]
                    voxel_size_for_volume = specific_structure["Voxel size for structure volume calc"]

                    structure_dimension_dict = specific_structure["Structure dimension at centroid dict"]
                    x_dim_len = structure_dimension_dict["X dimension length at centroid"]
                    y_dim_len = structure_dimension_dict["Y dimension length at centroid"]
                    z_dim_len = structure_dimension_dict["Z dimension length at centroid"]
                    voxel_size_for_dimension = specific_structure["Voxel size for structure dimension calc"]


                    structure_ID_list.append(structureID)
                    structure_type_list.append(struct_type)
                    structure_ref_num_list.append(structure_reference_number)
                    structure_max_pairwise_dist_list.append(maximum_distance)
                    structure_volume_list.append(structure_volume)
                    structure_index_list.append(specific_structure_index)
                    patient_ID_list.append(patientUID)
                    voxel_size_for_volume_list.append(voxel_size_for_volume)

                    structure_x_dim_list.append(x_dim_len)
                    structure_y_dim_list.append(y_dim_len)
                    structure_z_dim_list.append(z_dim_len)
                    voxel_size_for_dimension_list.append(voxel_size_for_dimension)


                else:
                    pass

        structure_info_dict_for_pandas_dataframe = {"Patient UID": patient_ID_list,
                                                    "Structure ID": structure_ID_list,
                                                    "Structure type": structure_type_list,
                                                    "Structure ref num": structure_ref_num_list,
                                                    "Structure index": structure_index_list,
                                                    "Structure max pair-wise distance": structure_max_pairwise_dist_list,
                                                    "Structure volume": structure_volume_list,
                                                    "Voxel side length (volume calculation)": voxel_size_for_volume_list,
                                                    "Structure X dim length (mm)": structure_x_dim_list,
                                                    "Structure Y dim length (mm)": structure_y_dim_list,
                                                    "Structure Z dim length (mm)": structure_z_dim_list,
                                                    "Voxel side length (dimension calculation)": voxel_size_for_dimension_list
                                                    }
        
        structure_info_pandas_data_frame = pandas.DataFrame.from_dict(data = structure_info_dict_for_pandas_dataframe)

        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Structure information (Non-BX)"] = structure_info_pandas_data_frame



def bx_structure_info_dataframe_builder(master_structure_ref_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref):
    

    for patientUID,pydicom_item in master_structure_ref_dict.items():
        patient_ID_list = []
        structure_ID_list = []
        structure_type_list = []
        structure_ref_num_list = []
        structure_index_list = []

        structure_volume_list = []
        structure_max_pairwise_dist_list = []
        voxel_size_for_volume_list = []

        nearest_dil_by_centroid_list = []
        nearest_dil_by_centroid_centroid_distance_list = []
        nearest_dil_by_centroid_surface_distance_list = []
        nearest_dil_by_surface_list = []
        nearest_dil_by_surface_centroid_distance_list = []
        nearest_dil_by_surface_surface_distance_list = []

        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                if structs == bx_ref:
                    structureID = specific_structure["ROI"]
                    structure_reference_number = specific_structure["Ref #"]
                    struct_type = structs

                    maximum_distance = specific_structure["Maximum pairwise distance"]
                    structure_volume = specific_structure["Structure volume"]
                    voxel_size_for_volume = specific_structure["Voxel size for structure volume calc"]

                    nearest_dils_by_surface_dict = specific_structure["Target DIL by surfaces dict"] 
                    nearest_dils_by_centroid_dict = specific_structure["Target DIL by centroid dict"]

                    patient_ID_list.append(patientUID)
                    structure_ID_list.append(structureID)
                    structure_type_list.append(struct_type)
                    structure_ref_num_list.append(structure_reference_number)
                    structure_index_list.append(specific_structure_index)

                    structure_volume_list.append(structure_volume)
                    structure_max_pairwise_dist_list.append(maximum_distance)
                    voxel_size_for_volume_list.append(voxel_size_for_volume)

                    structure_x_dim_list.append(x_dim_len)
                    structure_y_dim_list.append(y_dim_len)
                    structure_z_dim_list.append(z_dim_len)
                    voxel_size_for_dimension_list.append(voxel_size_for_dimension)


                else:
                    pass

        structure_info_dict_for_pandas_dataframe = {"Patient UID": patient_ID_list,
                                                    "Structure ID": structure_ID_list,
                                                    "Structure type": structure_type_list,
                                                    "Structure ref num": structure_ref_num_list,
                                                    "Structure index": structure_index_list,
                                                    "Structure max pair-wise distance": structure_max_pairwise_dist_list,
                                                    "Structure volume": structure_volume_list,
                                                    "Voxel side length (volume calculation)": voxel_size_for_volume_list,
                                                    "Structure X dim length (mm)": structure_x_dim_list,
                                                    "Structure Y dim length (mm)": structure_y_dim_list,
                                                    "Structure Z dim length (mm)": structure_z_dim_list,
                                                    "Voxel side length (dimension calculation)": voxel_size_for_dimension_list
                                                    }
        
        structure_info_pandas_data_frame = pandas.DataFrame.from_dict(data = structure_info_dict_for_pandas_dataframe)

        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Structure information (Non-BX)"] = structure_info_pandas_data_frame


def bx_nearest_dils_dataframe_builder(master_structure_reference_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref
                                       ):
    
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        sp_patient_relative_dil_dataframe_list = []
        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                if structs == bx_ref:
                    structureID = specific_structure["ROI"]
                    structure_reference_number = specific_structure["Ref #"]
                    bx_structure_info = (patientUID,
                                                structureID,
                                                bx_ref,
                                                structure_reference_number,
                                                specific_structure_index
                                                )

                
                    dil_distance_dict = specific_structure["Nearest DILs info dict"]


                    patientUID_list = []
                    structureID_list = []
                    bx_ref_list = []
                    structure_reference_number_list = []
                    specific_structure_index_list = []
                    dil_structureID_list = []
                    dil_ref_list = []
                    dil_structure_reference_number_list = []
                    specific_dil_structure_index_list = []
                    bx_centroid_vec_list = []
                    dil_centroid_vec_list =[]
                    vector_cent_to_cent_list = []
                    x_cent_to_cent_list = []
                    y_cent_to_cent_list = []
                    z_cent_to_cent_list = []
                    dist_cent_to_cent_list = []
                    nn_dist_surf_to_surf_list = []

                    
                    patientUID = bx_structure_info[0]
                    structureID = bx_structure_info[1]
                    bx_ref = bx_structure_info[2]
                    structure_reference_number = bx_structure_info[3]
                    specific_structure_index = bx_structure_info[4]
                                                                

                    for dil_structure_info, dil_distance_info in dil_distance_dict.items():

                        dil_structureID = dil_structure_info[0]
                        dil_ref = dil_structure_info[1]
                        dil_structure_reference_number = dil_structure_info[2]
                        specific_dil_structure_index = dil_structure_info[3]

                        bx_centroid_vec = dil_distance_info["Bx centroid vector"]
                        dil_centroid_vec = dil_distance_info["DIL centroid vector"]
                        vector_cent_to_cent = dil_distance_info["Vector DIL centroid - BX centroid"]
                        x_cent_to_cent = dil_distance_info["X to DIL centroid"]
                        y_cent_to_cent = dil_distance_info["Y to DIL centroid"]
                        z_cent_to_cent = dil_distance_info["Z to DIL centroid"]
                        dist_cent_to_cent = dil_distance_info["Distance DIL centroid - BX centroid"]
                        nn_dist_surf_to_surf = dil_distance_info["Shortest distance from BX surface to DIL surface"]


                        patientUID_list.append(patientUID)
                        structureID_list.append(structureID)
                        bx_ref_list.append(bx_ref)
                        structure_reference_number_list.append(structure_reference_number)
                        specific_structure_index_list.append(specific_structure_index)
                        dil_structureID_list.append(dil_structureID)
                        dil_ref_list.append(dil_ref)
                        dil_structure_reference_number_list.append(dil_structure_reference_number)
                        specific_dil_structure_index_list.append(specific_dil_structure_index)
                        bx_centroid_vec_list.append(bx_centroid_vec)
                        dil_centroid_vec_list.append(dil_centroid_vec)
                        vector_cent_to_cent_list.append(vector_cent_to_cent)
                        x_cent_to_cent_list.append(x_cent_to_cent)
                        y_cent_to_cent_list.append(y_cent_to_cent)
                        z_cent_to_cent_list.append(z_cent_to_cent)
                        dist_cent_to_cent_list.append(dist_cent_to_cent)
                        nn_dist_surf_to_surf_list.append(nn_dist_surf_to_surf)

                    else:
                        pass
                                                                
                    sp_bx_relative_dil_info_dict = {"Patient UID": patientUID_list,
                                                    "Structure ID": structureID_list,
                                                    "Struct type": bx_ref_list,
                                                    "Struct ref num": structure_reference_number_list,
                                                    "Structure index": specific_structure_index_list,
                                                    "Relative DIL ID": dil_structureID_list,
                                                    "Relative struct type": dil_ref_list,
                                                    "Relative DIL ref num": dil_structure_reference_number_list,
                                                    "Relative DIL index": specific_dil_structure_index_list,
                                                    "BX centroid vec": bx_centroid_vec_list,
                                                    "DIL centroid vec": dil_centroid_vec_list,
                                                    "BX to DIL centroid vector": vector_cent_to_cent_list,
                                                    "BX to DIL centroid (X)": x_cent_to_cent_list,
                                                    "BX to DIL centroid (Y)": y_cent_to_cent_list,
                                                    "BX to DIL centroid (Z)": z_cent_to_cent_list,
                                                    "BX to DIL centroid distance": dist_cent_to_cent_list,
                                                    "NN surface-surface distance": nn_dist_surf_to_surf_list
                                                    }
                    
                    sp_bx_relative_dil_dataframe = pandas.DataFrame.from_dict(data = sp_bx_relative_dil_info_dict)

                    sp_patient_relative_dil_dataframe_list.append(sp_bx_relative_dil_dataframe)

                    

        sp_patient_relative_dil_dataframe = pandas.concat(sp_patient_relative_dil_dataframe_list)

        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Nearest DILs info dataframe"] = sp_patient_relative_dil_dataframe




def bx_info_dataframe_builder(master_structure_reference_dict,
                            structs_referenced_list,
                            all_ref_key,
                            bx_ref,
                            target_dils_dataframe):
    print('test')




def dil_optimization_results_dataframe_builder(master_structure_reference_dict,
                                       all_ref_key,
                                       dil_ref
                                       ):
     
     for patientUID,pydicom_item in master_structure_reference_dict.items():
        optimal_locations_dataframe_list = []
        potential_optimal_locations_dataframe_list = []
        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):

            optimal_locations_dataframe = specific_dil_structure["Optimal biopsy location dataframe"]
            potential_optimal_locations_dataframe = specific_dil_structure["Optimal biopsy location (all latice points) dataframe"]

            optimal_locations_dataframe_list.append(optimal_locations_dataframe)
            potential_optimal_locations_dataframe_list.append(potential_optimal_locations_dataframe)

        sp_patient_optimal_dataframe = pandas.concat(optimal_locations_dataframe_list)
        sp_patient_potential_optimal_dataframe = pandas.concat(potential_optimal_locations_dataframe_list)

        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Optimal DIL targeting dataframe"] = sp_patient_optimal_dataframe
        pydicom_item[all_ref_key]["Multi-structure output data frames dict"]["Optimal DIL targeting entire lattice dataframe"] = sp_patient_potential_optimal_dataframe


