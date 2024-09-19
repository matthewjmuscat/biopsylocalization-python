import numpy as np
import pandas 
import csv
import math
import math_funcs as mf
import misc_tools
from pandas.api.types import union_categoricals
from scipy.stats import gaussian_kde
import math_funcs

def tissue_probability_dataframe_builder_by_bx_pt(patientUID,
                                                  specific_bx_structure,
                                                  specific_bx_structure_index,
                                         structure_miss_probability_roi,
                                         cancer_tissue_label,
                                         miss_structure_complement_label,
                                         biopsy_z_voxel_length
                                         ):
    
    bx_struct_roi = specific_bx_structure["ROI"]
    bx_struct_refnum = specific_bx_structure['Ref #']
    bx_simulated_bool = specific_bx_structure['Simulated bool']
    bx_type = specific_bx_structure["Simulated type"]
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
    X_point_wise_for_pd_data_frame_list = bx_points_bx_coords_sys_arr[:,0].tolist()
    Y_point_wise_for_pd_data_frame_list = bx_points_bx_coords_sys_arr[:,1].tolist()
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
        X_point_wise_for_pd_data_frame_list = X_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,0].tolist() 
        Y_point_wise_for_pd_data_frame_list = Y_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,1].tolist() 


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
    X_point_wise_for_pd_data_frame_list = X_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,0].tolist() 
    Y_point_wise_for_pd_data_frame_list = Y_point_wise_for_pd_data_frame_list + bx_points_bx_coords_sys_arr[:,1].tolist()
    binom_est_point_wise_for_pd_data_frame_list = binom_est_point_wise_for_pd_data_frame_list + miss_structure_binom_est_list
    std_err_point_wise_for_pd_data_frame_list = std_err_point_wise_for_pd_data_frame_list + miss_structure_standard_err_list
    binom_est_lower_CI_point_wise_for_pd_data_frame_list = binom_est_lower_CI_point_wise_for_pd_data_frame_list + miss_structure_CI_lower_list
    binom_est_upper_CI_point_wise_for_pd_data_frame_list = binom_est_upper_CI_point_wise_for_pd_data_frame_list + miss_structure_CI_upper_list
    nominal_point_wise_for_pd_data_frame_list = nominal_point_wise_for_pd_data_frame_list + miss_structure_nominal_list
        
    containment_output_dict_by_MC_trial_for_pandas_data_frame = {"Patient ID": patientUID,
                                                                "Bx structure ROI": bx_struct_roi,
                                                                "Bx refnum": bx_struct_refnum,
                                                                "Bx index": specific_bx_structure_index,
                                                                "Bx sim bool": bx_simulated_bool,
                                                                "Bx type": bx_type,
                                                                "Structure ROI": ROI_name_point_wise_for_pd_data_frame_list, 
                                                                "R (Bx frame)": pt_radius_point_wise_for_pd_data_frame_list,
                                                                "X (Bx frame)": X_point_wise_for_pd_data_frame_list, 
                                                                "Y (Bx frame)": Y_point_wise_for_pd_data_frame_list,  
                                                                "Z (Bx frame)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                "Mean probability (binom est)": binom_est_point_wise_for_pd_data_frame_list, 
                                                                "STD err": std_err_point_wise_for_pd_data_frame_list,
                                                                "Nominal containment": nominal_point_wise_for_pd_data_frame_list,
                                                                "CI lower vals": binom_est_lower_CI_point_wise_for_pd_data_frame_list,
                                                                "CI upper vals": binom_est_upper_CI_point_wise_for_pd_data_frame_list
                                                                }

    containment_output_by_MC_trial_pandas_data_frame = pandas.DataFrame.from_dict(data=containment_output_dict_by_MC_trial_for_pandas_data_frame)

    reference_dimension_col_name = "Z (Bx frame)"
    containment_output_by_MC_trial_pandas_data_frame = add_voxel_columns_helper_func(containment_output_by_MC_trial_pandas_data_frame, biopsy_z_voxel_length, reference_dimension_col_name)

    return containment_output_dict_by_MC_trial_for_pandas_data_frame, containment_output_by_MC_trial_pandas_data_frame


def cohort_and_multi_biopsy_mc_structure_specific_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key):
    cohort_mc_structure_specific_pt_wise_results_dataframe = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        multi_structure_mc_structure_specific_pt_wise_results_dataframe = pandas.DataFrame()
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):

            sp_bx_mc_structure_specific_pt_wise_results_dataframe = specific_bx_structure["MC data: compiled sim results dataframe"]

            multi_structure_mc_structure_specific_pt_wise_results_dataframe = pandas.concat([multi_structure_mc_structure_specific_pt_wise_results_dataframe,sp_bx_mc_structure_specific_pt_wise_results_dataframe]).reset_index(drop = True)

        multi_structure_mc_structure_specific_pt_wise_results_dataframe = convert_columns_to_categorical_and_downcast(multi_structure_mc_structure_specific_pt_wise_results_dataframe, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - Pt wise structure specific results"] = multi_structure_mc_structure_specific_pt_wise_results_dataframe

        cohort_mc_structure_specific_pt_wise_results_dataframe = pandas.concat([cohort_mc_structure_specific_pt_wise_results_dataframe,multi_structure_mc_structure_specific_pt_wise_results_dataframe]).reset_index(drop = True)

    cohort_mc_structure_specific_pt_wise_results_dataframe = convert_columns_to_categorical_and_downcast(cohort_mc_structure_specific_pt_wise_results_dataframe, threshold=0.25)

    return cohort_mc_structure_specific_pt_wise_results_dataframe


def cohort_and_multi_biopsy_mc_sum_to_one_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key):
    cohort_mc_sum_to_one_pt_wise_results_dataframe = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        multi_structure_mc_sum_to_one_pt_wise_results_dataframe = pandas.DataFrame()
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):

            mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = specific_bx_structure["MC data: compiled sim sum-to-one results dataframe"]

            multi_structure_mc_sum_to_one_pt_wise_results_dataframe = pandas.concat([multi_structure_mc_sum_to_one_pt_wise_results_dataframe,mc_compiled_results_sum_to_one_for_fixed_bx_dataframe]).reset_index(drop = True)

        multi_structure_mc_sum_to_one_pt_wise_results_dataframe = convert_columns_to_categorical_and_downcast(multi_structure_mc_sum_to_one_pt_wise_results_dataframe, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - sum-to-one mc results"] = multi_structure_mc_sum_to_one_pt_wise_results_dataframe

        cohort_mc_sum_to_one_pt_wise_results_dataframe = pandas.concat([cohort_mc_sum_to_one_pt_wise_results_dataframe,multi_structure_mc_sum_to_one_pt_wise_results_dataframe]).reset_index(drop = True)

    cohort_mc_sum_to_one_pt_wise_results_dataframe = convert_columns_to_categorical_and_downcast(cohort_mc_sum_to_one_pt_wise_results_dataframe, threshold=0.25)

    return cohort_mc_sum_to_one_pt_wise_results_dataframe


def cohort_and_multi_biopsy_mc_tissue_class_pt_wise_results_dataframe_builder(master_structure_reference_dict,
                                                                                    bx_ref,
                                                                                    all_ref_key):
    cohort_mc_tissue_class_pt_wise_results_dataframe = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        multi_structure_mc_tissue_class_pt_wise_results_dataframe = pandas.DataFrame()
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):

            sp_bx_mc_tissue_class_pt_wise_results_dataframe = specific_bx_structure["MC data: mutual compiled sim results dataframe"]

            multi_structure_mc_tissue_class_pt_wise_results_dataframe = pandas.concat([multi_structure_mc_tissue_class_pt_wise_results_dataframe,sp_bx_mc_tissue_class_pt_wise_results_dataframe]).reset_index(drop = True)
        
        multi_structure_mc_tissue_class_pt_wise_results_dataframe = convert_columns_to_categorical_and_downcast(multi_structure_mc_tissue_class_pt_wise_results_dataframe, threshold=0.25)
        
        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - Pt wise structure specific results"] = multi_structure_mc_tissue_class_pt_wise_results_dataframe

        cohort_mc_tissue_class_pt_wise_results_dataframe = pandas.concat([cohort_mc_tissue_class_pt_wise_results_dataframe,multi_structure_mc_tissue_class_pt_wise_results_dataframe]).reset_index(drop = True)

    cohort_mc_tissue_class_pt_wise_results_dataframe = convert_columns_to_categorical_and_downcast(cohort_mc_tissue_class_pt_wise_results_dataframe, threshold=0.25)

    return cohort_mc_tissue_class_pt_wise_results_dataframe


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


def global_scores_by_tissue_class_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key):
    
    cohort_global_tissue_class_dataframe = pandas.DataFrame()


    for patientUID,pydicom_item in master_structure_reference_dict.items():

        sp_patient_all_biopsies_global_containment_scores_by_tissue_class = pandas.DataFrame()

        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            
            bx_struct_roi = specific_bx_structure["ROI"]
            bx_struct_refnum = specific_bx_structure["Ref #"]
            num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
            simulated_bool = specific_bx_structure["Simulated bool"]
            bx_type = specific_bx_structure["Simulated type"]

            containment_output_by_bx_pt_pandas_data_frame = specific_bx_structure["Output data frames"]["Mutual containment output by bx point"] 
            
            # Note it is very important to convert grouping columns back to appropriate dtypes before grouping especially when grouping multiple columns simultaneously as this 
            # ensures that erronous grouping combinations are not produced!
            #containment_output_by_bx_pt_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_bx_pt_pandas_data_frame, ['Nominal containment'], [float])
            containment_output_by_bx_pt_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_bx_pt_pandas_data_frame, ['Nominal containment', 'Structure ROI'], [float, str])

            global_mean_binom_est_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Mean probability (binom est)'].mean()
            global_mean_binom_est_std_dev_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Mean probability (binom est)'].std()
            global_mean_binom_est_std_err_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Mean probability (binom est)'].sem()
            global_max_binom_est_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Mean probability (binom est)'].max()
            global_min_binom_est_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Mean probability (binom est)'].min()
            global_mean_nominal_containment_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Nominal containment'].mean()
            global_mean_nominal_containment_std_dev_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Nominal containment'].std()
            global_mean_nominal_containment_std_err_series = containment_output_by_bx_pt_pandas_data_frame.groupby(['Structure ROI'])['Nominal containment'].sem()
            
            sp_bx_global_containment_stats_dict = {"Patient ID": patientUID,
                              "Bx ID": bx_struct_roi,
                              "Bx refnum": bx_struct_refnum,
                              "Bx index": specific_bx_structure_index,
                              "Simulated bool": simulated_bool,
                              "Simulated type": bx_type,
                              'Global mean binom est': global_mean_binom_est_series, 
                              'Global max binom est': global_max_binom_est_series, 
                              'Global min binom est': global_min_binom_est_series, 
                              'Global standard deviation binom est': global_mean_binom_est_std_dev_series,
                              'Global standard error binom est': global_mean_binom_est_std_err_series,
                              "Global mean nominal": global_mean_nominal_containment_series,
                              "Global standard deviation nominal": global_mean_nominal_containment_std_dev_series,
                              "Global standard error nominal": global_mean_nominal_containment_std_err_series
                              }
            
            # Note that the reset_index(drop=False) line is crucial here because it keeps the structure ROI column from the groupby commands
            sp_bx_global_containment_stats_dataframe = pandas.DataFrame(sp_bx_global_containment_stats_dict).reset_index(drop = False) 

            sp_bx_global_containment_stats_dataframe[["Global CI 95 tuple binom est (lower)","Global CI 95 tuple binom est (upper)"]] = sp_bx_global_containment_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=('Global mean binom est', 'Global standard error binom est'), axis=1).tolist()
            sp_bx_global_containment_stats_dataframe[["Global CI 95 tuple nominal (lower)","Global CI 95 tuple nominal (upper)"]] = sp_bx_global_containment_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=("Global mean nominal", "Global standard error nominal"), axis=1).tolist()


            sp_patient_all_biopsies_global_containment_scores_by_tissue_class = pandas.concat([sp_patient_all_biopsies_global_containment_scores_by_tissue_class,sp_bx_global_containment_stats_dataframe], ignore_index = True)
            
            containment_output_by_bx_pt_pandas_data_frame = convert_columns_to_categorical_and_downcast(containment_output_by_bx_pt_pandas_data_frame, threshold=0.25)

            specific_bx_structure["Mutual containment output by bx point"] = containment_output_by_bx_pt_pandas_data_frame


        sp_patient_all_biopsies_global_containment_scores_by_tissue_class = convert_columns_to_categorical_and_downcast(sp_patient_all_biopsies_global_containment_scores_by_tissue_class, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - Global tissue class statistics"] = sp_patient_all_biopsies_global_containment_scores_by_tissue_class

        cohort_global_tissue_class_dataframe = pandas.concat([cohort_global_tissue_class_dataframe,sp_patient_all_biopsies_global_containment_scores_by_tissue_class], ignore_index = True)
    
    cohort_global_tissue_class_dataframe = convert_columns_to_categorical_and_downcast(cohort_global_tissue_class_dataframe, threshold=0.25)

    return cohort_global_tissue_class_dataframe




def global_scores_by_specific_structure_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key):
    
    cohort_global_tissue_structure_dataframe = pandas.DataFrame()


    for patientUID,pydicom_item in master_structure_reference_dict.items():

        sp_patient_all_biopsies_global_containment_scores_by_tissue_structure = pandas.DataFrame()

        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            
            bx_struct_roi = specific_bx_structure["ROI"]
            num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
            simulated_bool = specific_bx_structure["Simulated bool"]
            bx_type = specific_bx_structure["Simulated type"]
            bx_refnum = specific_bx_structure["Ref #"]

            containment_output_by_rel_structure_pandas_data_frame = specific_bx_structure["MC data: compiled sim results dataframe"] 

            # Note it is very important to convert grouping columns back to appropriate dtypes before grouping especially when grouping multiple columns simultaneously as this 
            # ensures that erronous grouping combinations are not produced!
            #containment_output_by_rel_structure_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_rel_structure_pandas_data_frame, ['Nominal'], [float])
            containment_output_by_rel_structure_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_rel_structure_pandas_data_frame, ['Nominal', "Relative structure index",'Relative structure ROI', 'Relative structure type'], [float, int, str, str])

            global_mean_binom_est_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Binomial estimator'].mean()
            global_mean_binom_est_std_dev_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Binomial estimator'].std()
            global_mean_binom_est_std_err_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Binomial estimator'].sem()
            global_max_binom_est_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Binomial estimator'].max()
            global_min_binom_est_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Binomial estimator'].min()
            global_mean_nominal_containment_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Nominal'].mean()
            global_mean_nominal_containment_std_dev_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Nominal'].std()
            global_mean_nominal_containment_std_err_series = containment_output_by_rel_structure_pandas_data_frame.groupby(['Relative structure ROI', 'Relative structure type', "Relative structure index"])['Nominal'].sem()
            
            sp_bx_global_containment_stats_dict = {"Patient ID": patientUID,
                              "Bx ID": bx_struct_roi,
                              "Bx index": specific_bx_structure_index,
                              "Bx refnum": bx_refnum,
                              "Simulated bool": simulated_bool,
                              "Simulated type": bx_type,
                              'Global mean binom est': global_mean_binom_est_series, 
                              'Global max binom est': global_max_binom_est_series, 
                              'Global min binom est': global_min_binom_est_series, 
                              'Global standard deviation binom est': global_mean_binom_est_std_dev_series,
                              'Global standard error binom est': global_mean_binom_est_std_err_series,
                              "Global mean nominal": global_mean_nominal_containment_series,
                              "Global standard deviation nominal": global_mean_nominal_containment_std_dev_series,
                              "Global standard error nominal": global_mean_nominal_containment_std_err_series
                              }
            
            # The reset_index(drop = False) line is crucial to retain the columns that are used in the groupby commands above
            sp_bx_global_containment_stats_dataframe = pandas.DataFrame(sp_bx_global_containment_stats_dict).reset_index(drop = False)

            sp_bx_global_containment_stats_dataframe[["Global CI 95 tuple binom est (lower)","Global CI 95 tuple binom est (upper)"]] = sp_bx_global_containment_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=('Global mean binom est', 'Global standard error binom est'), axis=1).tolist()
            sp_bx_global_containment_stats_dataframe[["Global CI 95 tuple nominal (lower)","Global CI 95 tuple nominal (upper)"]] = sp_bx_global_containment_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=("Global mean nominal", "Global standard error nominal"), axis=1).tolist()


            sp_patient_all_biopsies_global_containment_scores_by_tissue_structure = pandas.concat([sp_patient_all_biopsies_global_containment_scores_by_tissue_structure,sp_bx_global_containment_stats_dataframe], ignore_index = True)
            
            containment_output_by_rel_structure_pandas_data_frame = convert_columns_to_categorical_and_downcast(containment_output_by_rel_structure_pandas_data_frame, threshold=0.25)

            specific_bx_structure["MC data: compiled sim results dataframe"] = containment_output_by_rel_structure_pandas_data_frame


        # Move the Patient ID column to the beginning of the dataframe
        column_to_move = 'Patient ID'
        sp_patient_all_biopsies_global_containment_scores_by_tissue_structure = sp_patient_all_biopsies_global_containment_scores_by_tissue_structure[[column_to_move] + [col for col in sp_patient_all_biopsies_global_containment_scores_by_tissue_structure.columns if col != column_to_move]]

        sp_patient_all_biopsies_global_containment_scores_by_tissue_structure = convert_columns_to_categorical_and_downcast(sp_patient_all_biopsies_global_containment_scores_by_tissue_structure, threshold=0.25)


        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - Global tissue by structure statistics"] = sp_patient_all_biopsies_global_containment_scores_by_tissue_structure

        cohort_global_tissue_structure_dataframe = pandas.concat([cohort_global_tissue_structure_dataframe,sp_patient_all_biopsies_global_containment_scores_by_tissue_structure], ignore_index = True)
    
    cohort_global_tissue_structure_dataframe = convert_columns_to_categorical_and_downcast(cohort_global_tissue_structure_dataframe, threshold=0.25)


    return cohort_global_tissue_structure_dataframe



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




def tissue_length_threshold_dataframe_builder_NEW(master_structure_reference_dict,
                                                  bx_ref,
                                                  all_ref_key):
    
    for patientUID,pydicom_item in master_structure_reference_dict.items():

        sp_patient_DIL_issue_length_dataframe = pandas.DataFrame()

        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):

            bx_struct_roi = specific_bx_structure["ROI"]
            num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
            simulated_bool = specific_bx_structure["Simulated bool"]
            bx_type = specific_bx_structure["Simulated type"]
            
            tissue_length_by_threshold_dict = specific_bx_structure["MC data: tissue length above threshold dict"]

            sp_biopsy_DIL_issue_length_dataframe = pandas.DataFrame()

            for threshold, tissue_length_sp_treshold_dict in tissue_length_by_threshold_dict.items():

                num_bootstraps = tissue_length_sp_treshold_dict["Num bootstraps"]
                length_estimate_mean = tissue_length_sp_treshold_dict["Length estimate mean"]
                leangth_estimate_se = tissue_length_sp_treshold_dict["Length estimate se"]
                
                sp_threshold_global_tissue_length_stats_dict = {"Patient ID": [patientUID],
                                "Bx ID": [bx_struct_roi],
                                "Simulated bool": [simulated_bool],
                                "Simulated type": [bx_type],
                                'Probability threshold': [threshold], 
                                'Length estimate mean': [length_estimate_mean], 
                                'Length estimate SEM': [leangth_estimate_se]
                                }
                
                sp_threshold_global_tissue_length_stats_dataframe = pandas.DataFrame(sp_threshold_global_tissue_length_stats_dict)

                sp_biopsy_DIL_issue_length_dataframe = pandas.concat([sp_biopsy_DIL_issue_length_dataframe,sp_threshold_global_tissue_length_stats_dataframe])


            sp_patient_DIL_issue_length_dataframe = pandas.concat([sp_patient_DIL_issue_length_dataframe,sp_biopsy_DIL_issue_length_dataframe])


        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Tissue class - Tissue length above threshold"] = sp_patient_DIL_issue_length_dataframe.reset_index()


def tissue_volume_threshold_dataframe_builder_NEW(master_structure_reference_dict,
                                                  bx_ref):
    
    cohort_tissue_volume_above_threshold_dataframe = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():

        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):

            sp_bx_all_thresholds_volume_of_tissue_above_threshold_dataframe = specific_bx_structure["Output data frames"]["Tissue volume above threshold"]
            
            cohort_tissue_volume_above_threshold_dataframe = pandas.concat([cohort_tissue_volume_above_threshold_dataframe,sp_bx_all_thresholds_volume_of_tissue_above_threshold_dataframe], ignore_index = True)

            del specific_bx_structure["Output data frames"]["Tissue volume above threshold"]

    cohort_tissue_volume_above_threshold_dataframe = convert_columns_to_categorical_and_downcast(cohort_tissue_volume_above_threshold_dataframe, threshold=0.25)

    return cohort_tissue_volume_above_threshold_dataframe



def cohort_structure_features_dataframe_builder(master_structure_reference_dict,
                                                structs_referenced_list,
                                                bx_ref):
    cohort_structure_features_dataframe = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for structs in structs_referenced_list:
            if structs == bx_ref:
                continue
            else:
                for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                    sp_structure_shape_features_dataframe = specific_structure["Structure features dataframe"]

                    cohort_structure_features_dataframe = pandas.concat([cohort_structure_features_dataframe,sp_structure_shape_features_dataframe]).reset_index(drop = True)

    cohort_structure_features_dataframe = convert_columns_to_categorical_and_downcast(cohort_structure_features_dataframe, threshold=0.25)

    return cohort_structure_features_dataframe

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


def pointwise_mean_dose_and_standard_deviation_dataframe_builder(master_structure_reference_dict,
                                                                 bx_ref):
    # generate a pandas data frame that is used in numerous production plot functions
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):                        
            stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
            mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"].copy()
            std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"].copy()
            quantiles_dose_val_specific_bx_pt_dict_of_lists = stats_dose_val_all_MC_trials_by_bx_pt_list["Quantiles dose by bx pt dict"].copy()
            bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
            bx_points_XY_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr[:,0:2])
            pt_radius_bx_coord_sys = np.linalg.norm(bx_points_XY_bx_coords_sys_arr_list, axis = 1)

            dose_output_dict_for_pandas_data_frame = {"R (Bx frame)": pt_radius_bx_coord_sys, 
                                                        "Z (Bx frame)": bx_points_bx_coords_sys_arr[:,2], 
                                                        "Mean dose (Gy)": mean_dose_val_specific_bx_pt, 
                                                        "STD dose": std_dose_val_specific_bx_pt
                                                        }
            dose_output_dict_for_pandas_data_frame.update(quantiles_dose_val_specific_bx_pt_dict_of_lists)
            dose_output_pandas_data_frame = pandas.DataFrame(data=dose_output_dict_for_pandas_data_frame)
            
            dose_output_pandas_data_frame = convert_columns_to_categorical_and_downcast(dose_output_pandas_data_frame, threshold=0.25)


            specific_bx_structure["Output data frames"]["Dose output Z and radius"] = dose_output_pandas_data_frame
            #specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"] = dose_output_dict_for_pandas_data_frame

## DEPRECATED
def all_dose_data_by_trial_and_pt_from_MC_trial_dataframe_builder(master_structure_ref_dict,
                                                                  bx_ref
                                                                  ):
    cohort_all_dose_data_by_trial_and_pt = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            bx_structure_roi = specific_bx_structure["ROI"]
            dose_output_z_and_radius_dict_for_pandas_data_frame = specific_bx_structure["Output dicts for data frames"]["Dose output Z and radius"]
            pt_radius_bx_coord_sys = dose_output_z_and_radius_dict_for_pandas_data_frame["R (Bx frame)"]

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
            pt_index_list = []
            num_nominal_and_all_MC_trials = dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_arr.shape[1]
            for pt_index, specific_pt_all_MC_dose_vals in enumerate(dose_vals_nominal_and_all_MC_trials_by_sampled_bx_pt_list):
                pt_index_list = pt_index_list + [pt_index]*num_nominal_and_all_MC_trials
                pt_radius_point_wise_for_pd_data_frame_list = pt_radius_point_wise_for_pd_data_frame_list + [pt_radius_bx_coord_sys[pt_index]]*num_nominal_and_all_MC_trials
                axial_Z_point_wise_for_pd_data_frame_list = axial_Z_point_wise_for_pd_data_frame_list + [bx_points_bx_coords_sys_arr[pt_index,2]]*num_nominal_and_all_MC_trials
                dose_vals_point_wise_for_pd_data_frame_list = dose_vals_point_wise_for_pd_data_frame_list + specific_pt_all_MC_dose_vals
                MC_trial_index_point_wise_for_pd_data_frame_list = MC_trial_index_point_wise_for_pd_data_frame_list + list(range(0,num_nominal_and_all_MC_trials))
            
            # Note that the 0th MC trial num index is the nominal value
            dose_output_dict_by_MC_trial_for_pandas_data_frame = {"Patient ID": patientUID,
                                                                  "Bx ID": bx_structure_roi,
                                                                  "Bx index": specific_bx_structure_index,
                                                                  "Original pt index": pt_index_list,
                                                                "R (Bx frame)": pt_radius_point_wise_for_pd_data_frame_list, 
                                                                "Z (Bx frame)": axial_Z_point_wise_for_pd_data_frame_list, 
                                                                "Dose (Gy)": dose_vals_point_wise_for_pd_data_frame_list, 
                                                                "MC trial num": MC_trial_index_point_wise_for_pd_data_frame_list
                                                                }
            
            dose_output_nominal_and_all_MC_trials_pandas_data_frame = pandas.DataFrame.from_dict(data = dose_output_dict_by_MC_trial_for_pandas_data_frame)
            specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"] = dose_output_nominal_and_all_MC_trials_pandas_data_frame
            #specific_bx_structure["Output dicts for data frames"]["Point-wise dose output by MC trial number"] = dose_output_dict_by_MC_trial_for_pandas_data_frame

            cohort_all_dose_data_by_trial_and_pt = pandas.concat([cohort_all_dose_data_by_trial_and_pt,dose_output_nominal_and_all_MC_trials_pandas_data_frame]).reset_index(drop=True)

            del specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"]

    return cohort_all_dose_data_by_trial_and_pt




### WARNING VOXELIZED DOSE RESULTS DICT AND DICT OF LISTS IS NO LONGER USED!
def dose_output_voxelized_dataframe_builder(master_structure_ref_dict,
                                            bx_structs):
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            stats_dose_val_all_MC_trials_voxelized = specific_bx_structure["MC data: voxelized dose results dict (dict of lists)"]
            dose_vals_in_voxel = stats_dose_val_all_MC_trials_voxelized["All dose vals in voxel list"]
            z_range_of_voxel = stats_dose_val_all_MC_trials_voxelized["Voxel z range rounded"]

            max_points_in_voxel = max(len(x) for x in dose_vals_in_voxel)

            dose_output_voxelized_dict_for_pandas_data_frame = {str(z_range_of_voxel[i]): misc_tools.pad_or_truncate(dose_vals_in_voxel[i], max_points_in_voxel) for i in range(len(z_range_of_voxel))}
            dose_output_voxelized_pandas_data_frame = pandas.DataFrame(data=dose_output_voxelized_dict_for_pandas_data_frame)

            dose_output_voxelized_pandas_data_frame = convert_columns_to_categorical_and_downcast(dose_output_voxelized_pandas_data_frame, threshold=0.25)

            specific_bx_structure["Output data frames"]["Dose output voxelized"] = dose_output_voxelized_pandas_data_frame
            #specific_bx_structure["Output dicts for data frames"]["Dose output voxelized"] = dose_output_voxelized_dict_for_pandas_data_frame


def differential_dvh_dataframe_all_mc_trials_dataframe_builder(master_structure_ref_dict,
                                                            master_structure_info_dict,
                                                            bx_structs):
    
    num_MC_dose_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
    num_MC_dose_simulations_plus_nominal = num_MC_dose_simulations+1
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
            differential_dvh_histogram_percent_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
            differential_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
            differential_dvh_dose_vals_bin_centers = misc_tools.mean_of_adjacent_np(differential_dvh_dose_vals_by_MC_trial_1darr)
            differential_dvh_dose_vals_bin_centers_list = differential_dvh_dose_vals_bin_centers.tolist()
            differential_dvh_dose_vals_bin_widths = misc_tools.distance_between_neighbors_np(differential_dvh_dose_vals_by_MC_trial_1darr)
            differential_dvh_dose_vals_bin_widths_list = differential_dvh_dose_vals_bin_widths.tolist()
            differential_dvh_dose_vals_list = differential_dvh_dose_vals_by_MC_trial_1darr.tolist()
            differential_dvh_dose_bins_categorical_list = ['['+str(round(differential_dvh_dose_vals_list[i],1))+','+str(round(differential_dvh_dose_vals_list[i+1],1))+']' for i in range(len(differential_dvh_dose_vals_by_MC_trial_1darr)-1)]
            differential_dvh_histogram_percent_by_MC_trial_list_of_lists = differential_dvh_histogram_percent_by_MC_trial_arr.tolist()
            differential_dvh_bin_number_list = [i for i in differential_dvh_dose_bins_categorical_list]
            
            percent_vals_list = []
            dose_bins_list = differential_dvh_dose_bins_categorical_list*num_MC_dose_simulations_plus_nominal 
            dose_bin_centers_list = differential_dvh_dose_vals_bin_centers_list*num_MC_dose_simulations_plus_nominal
            dose_bin_widths_list = differential_dvh_dose_vals_bin_widths_list*num_MC_dose_simulations_plus_nominal
            dose_bin_number_list = differential_dvh_bin_number_list*num_MC_dose_simulations_plus_nominal
            mc_trial_index_list = []
            for mc_trial_index in range(num_MC_dose_simulations_plus_nominal):
                percent_vals_list = percent_vals_list + differential_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index]
                mc_trial_index_list = mc_trial_index_list + [mc_trial_index]*len(differential_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index])
            differential_dvh_dict_for_pandas_dataframe = {"Percent volume": percent_vals_list, 
                                                        "Dose bin (Gy)": dose_bins_list,
                                                        "Dose bin center (Gy)": dose_bin_centers_list,
                                                        "Dose bin width (Gy)": dose_bin_widths_list, 
                                                        "Dose bin number": dose_bin_number_list,
                                                        "MC trial": mc_trial_index_list}
            differential_dvh_pandas_dataframe = pandas.DataFrame.from_dict(differential_dvh_dict_for_pandas_dataframe)

            specific_bx_structure["Output data frames"]["Differential DVH by MC trial"] = differential_dvh_pandas_dataframe
            #specific_bx_structure["Output dicts for data frames"]["Differential DVH by MC trial"] = differential_dvh_dict_for_pandas_dataframe



def differential_dvh_dataframe_all_mc_trials_dataframe_builder_v2(master_structure_ref_dict, 
                                                                  master_structure_info_dict, 
                                                                  bx_structs):
    num_MC_dose_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
    num_MC_dose_simulations_plus_nominal = num_MC_dose_simulations + 1

    for patientUID, pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
            histogram_percent_arr = differential_dvh_dict["Percent arr"]
            dose_vals_edges_arr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
            
            # Calculate dose bin centers and widths using NumPy vectorized operations
            dose_bin_centers = (dose_vals_edges_arr[:-1] + dose_vals_edges_arr[1:]) / 2
            dose_bin_widths = np.diff(dose_vals_edges_arr)
            
            # Prepare data to fill DataFrame later
            percent_vals_list = []
            mc_trial_index_list = []
            dose_bin_left_edges = np.tile(dose_vals_edges_arr[:-1], num_MC_dose_simulations_plus_nominal)
            dose_bin_right_edges = np.tile(dose_vals_edges_arr[1:], num_MC_dose_simulations_plus_nominal)
            repeated_dose_bin_centers = np.tile(dose_bin_centers, num_MC_dose_simulations_plus_nominal)
            repeated_dose_bin_widths = np.tile(dose_bin_widths, num_MC_dose_simulations_plus_nominal)
            repeated_bin_numbers = np.tile(np.arange(len(dose_bin_centers)), num_MC_dose_simulations_plus_nominal)
            
            for mc_trial_index in range(num_MC_dose_simulations_plus_nominal):
                mc_histogram_percent = histogram_percent_arr[mc_trial_index]
                percent_vals_list.extend(mc_histogram_percent)
                mc_trial_index_list.extend([mc_trial_index] * len(mc_histogram_percent))
            
            # Creating the DataFrame from collected data
            differential_dvh_dict_for_pandas_dataframe = {
                "Percent volume": percent_vals_list,
                "Dose bin edge (left) (Gy)": dose_bin_left_edges,
                "Dose bin edge (right) (Gy)": dose_bin_right_edges,
                "Dose bin center (Gy)": repeated_dose_bin_centers,
                "Dose bin width (Gy)": repeated_dose_bin_widths,
                "Dose bin number": repeated_bin_numbers,
                "MC trial": mc_trial_index_list
            }
            differential_dvh_pandas_dataframe = pandas.DataFrame(differential_dvh_dict_for_pandas_dataframe)
            
            # Convert certain columns to categorical to save memory
            differential_dvh_pandas_dataframe = convert_columns_to_categorical_and_downcast(
                differential_dvh_pandas_dataframe, 
                threshold=0.25
            )
            
            
            specific_bx_structure["Output data frames"]["Differential DVH by MC trial"] = differential_dvh_pandas_dataframe






def cumulative_dvh_dataframe_all_mc_trials_dataframe_builder(master_structure_ref_dict,
                                                            master_structure_info_dict,
                                                            bx_structs):
    num_MC_dose_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
    num_MC_dose_simulations_plus_nominal = num_MC_dose_simulations + 1

    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            bx_struct_roi = specific_bx_structure["ROI"]
            # create cumulative DVH plots
            cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]
            cumulative_dvh_histogram_percent_by_MC_trial_arr = cumulative_dvh_dict["Percent arr"]
            cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]
            cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists = cumulative_dvh_histogram_percent_by_MC_trial_arr.tolist()
            cumulative_dvh_dose_vals_by_MC_trial_list = cumulative_dvh_dose_vals_by_MC_trial_1darr.tolist()
            percent_vals_list = []
            dose_vals_list = cumulative_dvh_dose_vals_by_MC_trial_list*num_MC_dose_simulations_plus_nominal 
            mc_trial_index_list = []
            for mc_trial_index in range(num_MC_dose_simulations_plus_nominal):
                percent_vals_list = percent_vals_list + cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index]
                mc_trial_index_list = mc_trial_index_list + [mc_trial_index]*len(cumulative_dvh_histogram_percent_by_MC_trial_list_of_lists[mc_trial_index])
            cumulative_dvh_dict_for_pandas_dataframe = {"Percent volume": percent_vals_list, 
                                                        "Dose (Gy)": dose_vals_list,
                                                        "MC trial": mc_trial_index_list}
            cumulative_dvh_pandas_dataframe = pandas.DataFrame.from_dict(cumulative_dvh_dict_for_pandas_dataframe)

            cumulative_dvh_pandas_dataframe = convert_columns_to_categorical_and_downcast(cumulative_dvh_pandas_dataframe, threshold=0.25)

            specific_bx_structure["Output data frames"]["Cumulative DVH by MC trial"] = cumulative_dvh_pandas_dataframe
            #specific_bx_structure["Output dicts for data frames"]["Cumulative DVH by MC trial"] = cumulative_dvh_dict_for_pandas_dataframe



def cumulative_dvh_dataframe_all_mc_trials_dataframe_builder_v2(master_structure_ref_dict, 
                                                                master_structure_info_dict, 
                                                                bx_structs):
    num_MC_dose_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
    num_MC_dose_simulations_plus_nominal = num_MC_dose_simulations + 1

    for patientUID, pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structs]):
            cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]
            percent_arr = cumulative_dvh_dict["Percent arr"]
            dose_vals_arr = cumulative_dvh_dict["Dose vals arr (Gy)"]

            # Assuming each MC trial has the same number of dose points
            num_dose_points = len(dose_vals_arr)  # Number of dose points in a single MC trial
            total_data_points = num_dose_points * num_MC_dose_simulations_plus_nominal

            # Preallocate arrays for all data
            percent_vals_array = np.empty(total_data_points)
            dose_vals_array = np.tile(dose_vals_arr, num_MC_dose_simulations_plus_nominal)
            mc_trial_index_array = np.repeat(np.arange(num_MC_dose_simulations_plus_nominal), num_dose_points)

            # Fill percent values array
            start_idx = 0
            for mc_trial_index in range(num_MC_dose_simulations_plus_nominal):
                end_idx = start_idx + num_dose_points
                percent_vals_array[start_idx:end_idx] = percent_arr[mc_trial_index]
                start_idx = end_idx
            
            # Creating the DataFrame from preallocated data
            cumulative_dvh_pandas_dataframe = pandas.DataFrame({
                "Percent volume": percent_vals_array,
                "Dose (Gy)": dose_vals_array,
                "MC trial": mc_trial_index_array
            })

            # Convert appropriate columns to categorical types to save memory
            cumulative_dvh_pandas_dataframe = convert_columns_to_categorical_and_downcast(cumulative_dvh_pandas_dataframe, threshold=0.25)


            # Store the DataFrame in the structure dictionary
            specific_bx_structure["Output data frames"]["Cumulative DVH by MC trial"] = cumulative_dvh_pandas_dataframe

















def cohort_creator_binom_est_by_pt_and_voxel_dataframe(master_structure_ref_dict,
                                                       bx_ref):
    cohort_all_binom_est_data_by_pt_and_voxel = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            containment_output_by_MC_trial_pandas_data_frame = specific_bx_structure["Output data frames"]["Mutual containment output by bx point"]
            
            cohort_all_binom_est_data_by_pt_and_voxel = pandas.concat([cohort_all_binom_est_data_by_pt_and_voxel,containment_output_by_MC_trial_pandas_data_frame], ignore_index = True)

            del specific_bx_structure["Output data frames"]["Mutual containment output by bx point"]

    cohort_all_binom_est_data_by_pt_and_voxel = convert_columns_to_categorical_and_downcast(cohort_all_binom_est_data_by_pt_and_voxel, threshold=0.25)

    return cohort_all_binom_est_data_by_pt_and_voxel

def all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_NEW(master_structure_ref_dict,
                                                                  bx_ref,
                                                                  biopsy_z_voxel_length
                                                                  ):

    cohort_all_dose_data_by_trial_and_pt = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            bx_structure_roi = specific_bx_structure["ROI"]
            bx_structure_refnum = specific_bx_structure["Ref #"]
            bx_structure_sim_bool = specific_bx_structure["Simulated bool"]
            bx_structure_sim_type = specific_bx_structure["Simulated type"]
            
            # Note that each row is a specific biopsy point, while the column is a particular MC trial
            dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"] 
            
            sp_bx_dose_distribution_all_trials_df = dose_NxD_array_to_dataframe_helper_function_v2(dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr)

            bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
            sp_bx_dose_distribution_all_trials_df = misc_tools.include_vector_columns_in_dataframe(sp_bx_dose_distribution_all_trials_df, 
                                                                                           bx_points_bx_coords_sys_arr, 
                                                                                           reference_column_name = 'Original pt index', 
                                                                                           new_column_name_x = "X (Bx frame)", 
                                                                                           new_column_name_y = "Y (Bx frame)", 
                                                                                           new_column_name_z = "Z (Bx frame)")
            

            # Add R column
            sp_bx_dose_distribution_all_trials_df["R (Bx frame)"] = np.sqrt(sp_bx_dose_distribution_all_trials_df['X (Bx frame)']**2 + sp_bx_dose_distribution_all_trials_df["Y (Bx frame)"]**2)
            
            # Add info columns in reverse order 
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Simulated bool", value=bx_structure_sim_bool)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Simulated type", value=bx_structure_sim_type)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Bx refnum", value=bx_structure_refnum)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Bx index", value=specific_bx_structure_index)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Bx ID", value=bx_structure_roi)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Patient ID", value=patientUID)


            # voxelize
            reference_dimension_col_name = "Z (Bx frame)"
            sp_bx_dose_distribution_all_trials_df = add_voxel_columns_helper_func(sp_bx_dose_distribution_all_trials_df, biopsy_z_voxel_length, reference_dimension_col_name)

            
            #specific_bx_structure["Output dicts for data frames"]["Point-wise dose output by MC trial number"] = differential_dvh_dict_for_pandas_dataframe

            ### SAVE A TREMENDOUS AMOUNT OF MEMORY BY CONVERTING COLUMNS TO CATEGORICALS!
            ### BUT UNFORTUNATELY ITS TOO SLOW
            cohort_all_dose_data_by_trial_and_pt = pandas.concat([cohort_all_dose_data_by_trial_and_pt,sp_bx_dose_distribution_all_trials_df]).reset_index(drop=True)
            #cat_columns = ["Patient ID","Bx ID","Bx index", "Bx refnum", "Simulated type", "Simulated bool", "Original pt index", "MC trial num", "Voxel index"]
            #cohort_all_dose_data_by_trial_and_pt = concatenate_with_auto_categoricals(cohort_all_dose_data_by_trial_and_pt, sp_bx_dose_distribution_all_trials_df, threshold=0.25)

            sp_bx_dose_distribution_all_trials_df = convert_columns_to_categorical_and_downcast(sp_bx_dose_distribution_all_trials_df, threshold=0.25)
            specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"] = sp_bx_dose_distribution_all_trials_df

    cohort_all_dose_data_by_trial_and_pt = convert_columns_to_categorical_and_downcast(cohort_all_dose_data_by_trial_and_pt, threshold=0.25)
    
    return cohort_all_dose_data_by_trial_and_pt


# cohort dataframe for this is too big!
def all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_NEW_no_cohort(master_structure_ref_dict,
                                                                  bx_ref,
                                                                  biopsy_z_voxel_length,
                                                                  all_ref
                                                                  ):
    
    #cohort_all_dose_data_by_trial_and_pt = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_ref_dict.items():
        #sp_patient_all_dose_data_by_trial_and_pt = pandas.DataFrame()
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            bx_structure_roi = specific_bx_structure["ROI"]
            bx_structure_refnum = specific_bx_structure["Ref #"]
            bx_structure_sim_bool = specific_bx_structure["Simulated bool"]
            bx_structure_sim_type = specific_bx_structure["Simulated type"]
            
            # Note that each row is a specific biopsy point, while the column is a particular MC trial
            dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"] 
            
            sp_bx_dose_distribution_all_trials_df = dose_NxD_array_to_dataframe_helper_function_v2(dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr)

            bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
            sp_bx_dose_distribution_all_trials_df = misc_tools.include_vector_columns_in_dataframe(sp_bx_dose_distribution_all_trials_df, 
                                                                                           bx_points_bx_coords_sys_arr, 
                                                                                           reference_column_name = 'Original pt index', 
                                                                                           new_column_name_x = "X (Bx frame)", 
                                                                                           new_column_name_y = "Y (Bx frame)", 
                                                                                           new_column_name_z = "Z (Bx frame)")
            

            # Add R column
            sp_bx_dose_distribution_all_trials_df["R (Bx frame)"] = np.sqrt(sp_bx_dose_distribution_all_trials_df['X (Bx frame)']**2 + sp_bx_dose_distribution_all_trials_df["Y (Bx frame)"]**2)
            
            # Add info columns in reverse order 
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Simulated bool", value=bx_structure_sim_bool)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Simulated type", value=bx_structure_sim_type)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Bx refnum", value=bx_structure_refnum)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Bx index", value=specific_bx_structure_index)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Bx ID", value=bx_structure_roi)
            sp_bx_dose_distribution_all_trials_df.insert(loc=0, column="Patient ID", value=patientUID)


            # voxelize
            reference_dimension_col_name = "Z (Bx frame)"
            sp_bx_dose_distribution_all_trials_df = add_voxel_columns_helper_func(sp_bx_dose_distribution_all_trials_df, biopsy_z_voxel_length, reference_dimension_col_name)

            #sp_patient_all_dose_data_by_trial_and_pt = pandas.concat([sp_patient_all_dose_data_by_trial_and_pt,sp_bx_dose_distribution_all_trials_df], ignore_index = True)

            sp_bx_dose_distribution_all_trials_df = convert_columns_to_categorical_and_downcast(sp_bx_dose_distribution_all_trials_df, threshold=0.25)
            specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"] = sp_bx_dose_distribution_all_trials_df
            #specific_bx_structure["Output dicts for data frames"]["Point-wise dose output by MC trial number"] = differential_dvh_dict_for_pandas_dataframe

        #sp_patient_all_dose_data_by_trial_and_pt = convert_columns_to_categorical_and_downcast(sp_patient_all_dose_data_by_trial_and_pt, threshold=0.25)

        #pydicom_item[all_ref]["Dosimetry - All points and trials"] = sp_patient_all_dose_data_by_trial_and_pt



def all_dose_data_by_trial_and_pt_from_dataframe_builder_and_voxelizer_v4(master_structure_ref_dict, bx_ref, biopsy_z_voxel_length, all_ref):
    for patientUID, pydicom_item in master_structure_ref_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            # Extract relevant information
            bx_structure_roi = specific_bx_structure["ROI"]
            bx_structure_refnum = specific_bx_structure["Ref #"]
            bx_structure_sim_bool = specific_bx_structure["Simulated bool"]
            bx_structure_sim_type = specific_bx_structure["Simulated type"]
            dose_vals_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"]

            # Convert the dose values array into a DataFrame
            sp_bx_dose_distribution_all_trials_df = dose_NxD_array_to_dataframe_helper_function_v2(dose_vals_arr)
            
            # Include coordinate columns using a helper function
            bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
            sp_bx_dose_distribution_all_trials_df = misc_tools.include_vector_columns_in_dataframe(
                sp_bx_dose_distribution_all_trials_df, 
                bx_coords_sys_arr, 
                'Original pt index', 
                'X (Bx frame)', 
                'Y (Bx frame)', 
                'Z (Bx frame)'
            )

            # Compute the radial distance for each point
            sp_bx_dose_distribution_all_trials_df["R (Bx frame)"] = np.sqrt(
                sp_bx_dose_distribution_all_trials_df['X (Bx frame)']**2 + 
                sp_bx_dose_distribution_all_trials_df["Y (Bx frame)"]**2
            )

            # Add identifying and categorical information
            sp_bx_dose_distribution_all_trials_df["Simulated bool"] = bx_structure_sim_bool
            sp_bx_dose_distribution_all_trials_df["Simulated type"] = bx_structure_sim_type
            sp_bx_dose_distribution_all_trials_df["Bx refnum"] = bx_structure_refnum
            sp_bx_dose_distribution_all_trials_df["Bx index"] = specific_bx_structure_index
            sp_bx_dose_distribution_all_trials_df["Bx ID"] = bx_structure_roi
            sp_bx_dose_distribution_all_trials_df["Patient ID"] = patientUID

            # Add voxel information
            sp_bx_dose_distribution_all_trials_df = add_voxel_columns_helper_func(
                sp_bx_dose_distribution_all_trials_df, 
                biopsy_z_voxel_length, 
                "Z (Bx frame)"
            )

            # Convert certain columns to categorical to save memory
            sp_bx_dose_distribution_all_trials_df = convert_columns_to_categorical_and_downcast(
                sp_bx_dose_distribution_all_trials_df, 
                threshold=0.25
            )

            # Store the updated DataFrame in the structure dictionary
            specific_bx_structure["Output data frames"]["Point-wise dose output by MC trial number"] = sp_bx_dose_distribution_all_trials_df




### THIS FUNCTION HELPS THE ABOVE DOSE DISTRIBUTION CREATOR DF BY TURNING THE NXD ARRAY OF DOSE VALS FROM MC SIM INTO A DATAFRAME!
def dose_NxD_array_to_dataframe_helper_function(arr):
    # Preparing lists to hold data
    original_pt_index = []
    dose_gy = []
    mc_trial_num = []
    
    # Iterate over the array to fill the lists
    for i in range(arr.shape[0]):  # For each row
        for j in range(arr.shape[1]):  # For each column
            original_pt_index.append(i)
            dose_gy.append(arr[i, j])
            mc_trial_num.append(j)
    
    # Create DataFrame
    df = pandas.DataFrame({
        "Original pt index": original_pt_index,
        "Dose (Gy)": dose_gy,
        "MC trial num": mc_trial_num
    })
    
    return df

### THIS ONE IS WAY FASTER!
def dose_NxD_array_to_dataframe_helper_function_v2(arr):
    n_rows, n_cols = arr.shape
    original_pt_index = np.repeat(np.arange(n_rows), n_cols)
    dose_gy = arr.ravel()  # Flatten the array to a 1D array directly
    mc_trial_num = np.tile(np.arange(n_cols), n_rows)

    # Create DataFrame directly with preallocated arrays
    df = pandas.DataFrame({
        "Original pt index": original_pt_index,
        "Dose (Gy)": dose_gy,
        "MC trial num": mc_trial_num
    })
    
    return df

### THIS FUNCTION HELPS THE ABOVE DOSE DSITRIBUTION CREATOR DF BY ADDING VOXEL COLUMNS TO THE DATAFRAME!
def add_voxel_columns_helper_func(df, biopsy_z_voxel_length, reference_dimension_col_name):
    # Assuming 'reference_dimension_col_name' column exists in the DataFrame and biopsy_z_voxel_length is a positive float
    df['Voxel index'] = (df[reference_dimension_col_name] // biopsy_z_voxel_length) + 1
    df['Voxel index'] = df['Voxel index'].astype(int)
    df['Voxel begin (Z)'] = (df['Voxel index'] - 1) * biopsy_z_voxel_length
    df['Voxel end (Z)'] = df['Voxel begin (Z)'] + biopsy_z_voxel_length

    # Adjust the last voxel's end if it exceeds the maximum 'Z (Bx frame)' value
    max_z = df[reference_dimension_col_name].max()
    df.loc[df['Voxel end (Z)'] > max_z, 'Voxel end (Z)'] = max_z

    return df





def global_dosimetry_values_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key):
    
    cohort_global_dosimetry_dataframe = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():

        sp_patient_all_biopsies_global_dosimetry = pandas.DataFrame()

        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            
            bx_struct_roi = specific_bx_structure["ROI"]
            num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
            simulated_bool = specific_bx_structure["Simulated bool"]
            bx_type = specific_bx_structure["Simulated type"]
            bx_refnum = specific_bx_structure["Ref #"]

            sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame = specific_bx_structure['Output data frames']['Point-wise dose output by MC trial number'] 

            # Note it is very important to convert grouping columns back to appropriate dtypes before grouping especially when grouping multiple columns simultaneously as this 
            # ensures that erronous grouping combinations are not produced!
            #containment_output_by_rel_structure_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_rel_structure_pandas_data_frame, ['Nominal'], [float])
            sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame = misc_tools.convert_categorical_columns(sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame, ['Dose (Gy)'], [float])
            sp_bx_point_wise_dose_output_nominal_pandas_data_frame = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame[sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['MC trial num'] == 0]

            global_mean_dose_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].mean()
            global_dose_std_dev_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].std()
            global_dose_std_err_in_mean_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].sem()
            global_max_dose_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].max()
            global_min_dose_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].min()
            global_min_dose_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].min()
            global_quantiles_dose_series = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'].quantile([0.05,0.25,0.5,0.75,0.95])
            
            global_nominal_mean_dose_series = sp_bx_point_wise_dose_output_nominal_pandas_data_frame['Dose (Gy)'].mean()
            global_nominal_dose_std_dev_series = sp_bx_point_wise_dose_output_nominal_pandas_data_frame['Dose (Gy)'].std()
            global_nominal_dose_std_err_in_mean_series = sp_bx_point_wise_dose_output_nominal_pandas_data_frame['Dose (Gy)'].sem()
            global_nominal_max_dose_series = sp_bx_point_wise_dose_output_nominal_pandas_data_frame['Dose (Gy)'].max()
            global_nominal_min_dose_series = sp_bx_point_wise_dose_output_nominal_pandas_data_frame['Dose (Gy)'].min()

            global_max_density_dose = math_funcs.find_max_kde_dose(sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['Dose (Gy)'], num_eval_pts = 1000)

            global_nominal_max_density_dose = math_funcs.find_max_kde_dose(sp_bx_point_wise_dose_output_nominal_pandas_data_frame['Dose (Gy)'], num_eval_pts = 1000)
            
            sp_bx_global_dose_stats_dict = {"Patient ID": patientUID,
                                            "Bx ID": bx_struct_roi,
                                            "Bx index": specific_bx_structure_index,
                                            "Bx refnum": bx_refnum,
                                            "Simulated bool": simulated_bool,
                                            "Simulated type": bx_type,
                                            'Global max density dose': global_max_density_dose,
                                            'Global mean dose': global_mean_dose_series, 
                                            'Global max dose': global_max_dose_series, 
                                            'Global min dose': global_min_dose_series, 
                                            'Global standard deviation dose': global_dose_std_dev_series,
                                            'Global standard error dose': global_dose_std_err_in_mean_series,
                                            'Global q05 dose': global_quantiles_dose_series[0.05],
                                            'Global q25 dose': global_quantiles_dose_series[0.25],
                                            'Global q50 dose': global_quantiles_dose_series[0.5],
                                            'Global q75 dose': global_quantiles_dose_series[0.75],
                                            'Global q95 dose': global_quantiles_dose_series[0.95],
                                            'Global nominal max density dose': global_nominal_max_density_dose,
                                            'Global nominal mean dose': global_nominal_mean_dose_series, 
                                            'Global nominal max dose': global_nominal_max_dose_series, 
                                            'Global nominal min dose': global_nominal_min_dose_series, 
                                            'Global nominal standard deviation dose': global_nominal_dose_std_dev_series,
                                            'Global nominal standard error dose': global_nominal_dose_std_err_in_mean_series,
                                            }
            
            sp_bx_global_dose_stats_dataframe = pandas.DataFrame(sp_bx_global_dose_stats_dict, index=[0])

            sp_bx_global_dose_stats_dataframe[["Global CI 95 tuple dose (lower)","Global CI 95 tuple dose (upper)"]] = sp_bx_global_dose_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=('Global mean dose', 'Global standard error dose'), axis=1).tolist()
            sp_bx_global_dose_stats_dataframe[["Global CI 95 tuple nominal dose (lower)","Global CI 95 tuple nominal dose (upper)"]] = sp_bx_global_dose_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=('Global nominal mean dose', 'Global nominal standard error dose'), axis=1).tolist()


            sp_patient_all_biopsies_global_dosimetry = pandas.concat([sp_patient_all_biopsies_global_dosimetry,sp_bx_global_dose_stats_dataframe], ignore_index = True)
            
            sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame = convert_columns_to_categorical_and_downcast(sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame, threshold=0.25)

            specific_bx_structure['Output data frames']['Point-wise dose output by MC trial number']  = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame

        sp_patient_all_biopsies_global_dosimetry = convert_columns_to_categorical_and_downcast(sp_patient_all_biopsies_global_dosimetry, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Dosimetry - Global dosimetry statistics"] = sp_patient_all_biopsies_global_dosimetry

        cohort_global_dosimetry_dataframe = pandas.concat([cohort_global_dosimetry_dataframe,sp_patient_all_biopsies_global_dosimetry], ignore_index = True)
    
    cohort_global_dosimetry_dataframe = convert_columns_to_categorical_and_downcast(cohort_global_dosimetry_dataframe, threshold=0.25)

    return cohort_global_dosimetry_dataframe




def global_dosimetry_by_voxel_values_dataframe_builder(master_structure_reference_dict,
                                                    bx_ref,
                                                    all_ref_key):
    
    cohort_global_dosimetry_dataframe = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():

        sp_patient_all_biopsies_global_dosimetry = pandas.DataFrame()

        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            
            bx_struct_roi = specific_bx_structure["ROI"]
            num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
            simulated_bool = specific_bx_structure["Simulated bool"]
            bx_type = specific_bx_structure["Simulated type"]
            bx_refnum = specific_bx_structure["Ref #"]

            sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame = specific_bx_structure['Output data frames']['Point-wise dose output by MC trial number'] 

            # Note it is very important to convert grouping columns back to appropriate dtypes before grouping especially when grouping multiple columns simultaneously as this 
            # ensures that erronous grouping combinations are not produced!
            #containment_output_by_rel_structure_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_rel_structure_pandas_data_frame, ['Nominal'], [float])
            sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame = misc_tools.convert_categorical_columns(sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame, ['Dose (Gy)', 'Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'], [float, int, float, float])
            sp_bx_point_wise_dose_output_nominal_pandas_data_frame = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame[sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame['MC trial num'] == 0]
            
            sp_bx_global_grouped_df = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame.groupby(['Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'])
            sp_bx_nominal_global_grouped_df = sp_bx_point_wise_dose_output_nominal_pandas_data_frame.groupby(['Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'])

            global_by_voxel_mean_dose_series = sp_bx_global_grouped_df['Dose (Gy)'].mean()
            global_by_voxel_dose_std_dev_series = sp_bx_global_grouped_df['Dose (Gy)'].std()
            global_by_voxel_dose_std_err_in_mean_series = sp_bx_global_grouped_df['Dose (Gy)'].sem()
            global_by_voxel_max_dose_series = sp_bx_global_grouped_df['Dose (Gy)'].max()
            global_by_voxel_min_dose_series = sp_bx_global_grouped_df['Dose (Gy)'].min()
            global_by_voxel_min_dose_series = sp_bx_global_grouped_df['Dose (Gy)'].min()
            global_by_voxel_quantiles_dose_series = sp_bx_global_grouped_df['Dose (Gy)'].quantile([0.05,0.25,0.5,0.75,0.95])
            global_by_voxel_quantiles_dose_series_unstacked = global_by_voxel_quantiles_dose_series.unstack()

            
            global_by_voxel_nominal_mean_dose_series = sp_bx_nominal_global_grouped_df['Dose (Gy)'].mean()
            global_by_voxel_nominal_dose_std_dev_series = sp_bx_nominal_global_grouped_df['Dose (Gy)'].std()
            global_by_voxel_nominal_dose_std_err_in_mean_series = sp_bx_nominal_global_grouped_df['Dose (Gy)'].sem()
            global_by_voxel_nominal_max_dose_series = sp_bx_nominal_global_grouped_df['Dose (Gy)'].max()
            global_by_voxel_nominal_min_dose_series = sp_bx_nominal_global_grouped_df['Dose (Gy)'].min()
            
            global_by_voxel_max_density_dose_series = sp_bx_global_grouped_df['Dose (Gy)'].apply(math_funcs.find_max_kde_dose, num_eval_pts=1000)

            sp_bx_global_dose_stats_dict = {"Patient ID": patientUID,
                                            "Bx ID": bx_struct_roi,
                                            "Bx index": specific_bx_structure_index,
                                            "Bx refnum": bx_refnum,
                                            "Simulated bool": simulated_bool,
                                            "Simulated type": bx_type,
                                            'Global max density dose': global_by_voxel_max_density_dose_series,
                                            'Global mean dose': global_by_voxel_mean_dose_series, 
                                            'Global max dose': global_by_voxel_max_dose_series, 
                                            'Global min dose': global_by_voxel_min_dose_series, 
                                            'Global standard deviation dose': global_by_voxel_dose_std_dev_series,
                                            'Global standard error dose': global_by_voxel_dose_std_err_in_mean_series,
                                            'Global q05 dose': global_by_voxel_quantiles_dose_series_unstacked[0.05],
                                            'Global q25 dose': global_by_voxel_quantiles_dose_series_unstacked[0.25],
                                            'Global q50 dose': global_by_voxel_quantiles_dose_series_unstacked[0.5],
                                            'Global q75 dose': global_by_voxel_quantiles_dose_series_unstacked[0.75],
                                            'Global q95 dose': global_by_voxel_quantiles_dose_series_unstacked[0.95],
                                            'Global nominal mean dose': global_by_voxel_nominal_mean_dose_series, 
                                            'Global nominal max dose': global_by_voxel_nominal_max_dose_series, 
                                            'Global nominal min dose': global_by_voxel_nominal_min_dose_series, 
                                            'Global nominal standard deviation dose': global_by_voxel_nominal_dose_std_dev_series,
                                            'Global nominal standard error dose': global_by_voxel_nominal_dose_std_err_in_mean_series,
                                            }
            
            # the reset_index(drop=False) method is crucial to maintain the voxel index column which was used as a grouping column above
            sp_bx_global_dose_stats_dataframe = pandas.DataFrame(sp_bx_global_dose_stats_dict).reset_index(drop=False)

            sp_bx_global_dose_stats_dataframe[["Global CI 95 tuple dose (lower)","Global CI 95 tuple dose (upper)"]] = sp_bx_global_dose_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=('Global mean dose', 'Global standard error dose'), axis=1).tolist()
            sp_bx_global_dose_stats_dataframe[["Global CI 95 tuple nominal dose (lower)","Global CI 95 tuple nominal dose (upper)"]] = sp_bx_global_dose_stats_dataframe.apply(normal_CI_estimator_by_dataframe_row, args=('Global nominal mean dose', 'Global nominal standard error dose'), axis=1).tolist()


            sp_patient_all_biopsies_global_dosimetry = pandas.concat([sp_patient_all_biopsies_global_dosimetry,sp_bx_global_dose_stats_dataframe], ignore_index = True)
            
            sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame = convert_columns_to_categorical_and_downcast(sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame, threshold=0.25)

            specific_bx_structure['Output data frames']['Point-wise dose output by MC trial number']  = sp_bx_point_wise_dose_output_by_mc_trial_pandas_data_frame

        sp_patient_all_biopsies_global_dosimetry = convert_columns_to_categorical_and_downcast(sp_patient_all_biopsies_global_dosimetry, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["Dosimetry - Global dosimetry by voxel statistics"] = sp_patient_all_biopsies_global_dosimetry

        cohort_global_dosimetry_dataframe = pandas.concat([cohort_global_dosimetry_dataframe,sp_patient_all_biopsies_global_dosimetry], ignore_index = True)
    
    cohort_global_dosimetry_dataframe = convert_columns_to_categorical_and_downcast(cohort_global_dosimetry_dataframe, threshold=0.25)

    return cohort_global_dosimetry_dataframe
























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

        structure_info_dict_for_pandas_dataframe = {"Patient ID": patient_ID_list,
                                                    "Structure ID": structure_ID_list,
                                                    "Structure type": structure_type_list,
                                                    "Structure ref num": structure_ref_num_list,
                                                    "Structure index": structure_index_list,
                                                    "Structure max pair-wise distance": structure_max_pairwise_dist_list,
                                                    "Structure volume": structure_volume_list,
                                                    "Voxel side length": voxel_size_list
                                                    }
        
        structure_info_pandas_data_frame = pandas.DataFrame.from_dict(data = structure_info_dict_for_pandas_dataframe)

        structure_info_pandas_data_frame = convert_columns_to_categorical_and_downcast(structure_info_pandas_data_frame, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Structure information"] = structure_info_pandas_data_frame



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

        structure_info_dict_for_pandas_dataframe = {"Patient ID": patient_ID_list,
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

        structure_info_pandas_data_frame = convert_columns_to_categorical_and_downcast(structure_info_pandas_data_frame, threshold=0.25)
        
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Structure information dimension"] = structure_info_pandas_data_frame



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

        structure_info_dict_for_pandas_dataframe = {"Patient ID": patient_ID_list,
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

        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Structure information (Non-BX)"] = structure_info_pandas_data_frame





def bx_nearest_dils_dataframe_builder(master_structure_reference_dict,
                                       structs_referenced_list,
                                       all_ref_key,
                                       bx_ref
                                       ):
    
    cohort_nearest_dils_dataframe = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        sp_patient_relative_dil_dataframe_list = []
        for structs in structs_referenced_list:
            for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                if structs == bx_ref:
                    structureID = specific_structure["ROI"]
                    structure_reference_number = specific_structure["Ref #"]
                    bx_simulated_type = specific_structure["Simulated type"]
                    bx_simulated_bool = specific_structure["Simulated bool"]
                    bx_structure_info = (patientUID,
                                                structureID,
                                                bx_ref,
                                                structure_reference_number,
                                                specific_structure_index
                                                )

                
                    dil_distance_dict = specific_structure["Nearest DILs info dict"]
                    target_dil_by_centroid_dict = specific_structure['Target DIL by centroid dict']
                    target_dil_by_surfaces_dict = specific_structure['Target DIL by surfaces dict']
                    bx_location_in_prostate_dict = specific_structure['Bx location in prostate dict']
                    bx_AP_LR_SI_location_in_prostate_dict = bx_location_in_prostate_dict['Bx position in prostate']
                    bx_relative_prostate_info_dict = bx_location_in_prostate_dict['Relative prostate info']
                    bx_relative_prostate_id = bx_relative_prostate_info_dict['Structure ID']
                    bx_relative_prostate_index = bx_relative_prostate_info_dict['Index number']
                    bx_relative_prostate_struct_type = bx_relative_prostate_info_dict['Struct ref type']
                    bx_LR_pos = bx_AP_LR_SI_location_in_prostate_dict['LR']
                    bx_AP_pos = bx_AP_LR_SI_location_in_prostate_dict['AP']
                    bx_SI_pos = bx_AP_LR_SI_location_in_prostate_dict['SI']




                    patientUID_list = []
                    structureID_list = []
                    sim_type_list = []
                    sim_bool_list = []
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
                    x_dil_centroid_frame_list = []
                    y_dil_centroid_frame_list = []
                    z_dil_centroid_frame_list = []
                    dist_cent_to_cent_list = []
                    nn_dist_surf_to_surf_list = []
                    target_dil_by_centroids_list = []
                    target_dil_by_surfaces_list = []
                    bx_relative_reference_prostate_structure_type_list = []
                    bx_relative_reference_prostate_id_list = []
                    bx_relative_reference_prostate_index_list = []
                    bx_position_in_prostate_LR_list = []
                    bx_position_in_prostate_AP_list = []
                    bx_position_in_prostate_SI_list = []
                    
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

                        # using tuples instead of storing numpy arrays in dataframe cells because numpy arrays are unhashable!
                        bx_centroid_vec = tuple(dil_distance_info["Bx centroid vector"])
                        dil_centroid_vec = tuple(dil_distance_info["DIL centroid vector"])
                        vector_cent_to_cent = tuple(dil_distance_info["Vector DIL centroid - BX centroid"])
                        x_cent_to_cent = dil_distance_info["X to DIL centroid"]
                        y_cent_to_cent = dil_distance_info["Y to DIL centroid"]
                        z_cent_to_cent = dil_distance_info["Z to DIL centroid"]
                        x_dil_centroid_frame = -dil_distance_info["X to DIL centroid"]
                        y_dil_centroid_frame = -dil_distance_info["Y to DIL centroid"]
                        z_dil_centroid_frame = -dil_distance_info["Z to DIL centroid"]
                        dist_cent_to_cent = dil_distance_info["Distance DIL centroid - BX centroid"]
                        nn_dist_surf_to_surf = dil_distance_info["Shortest distance from BX surface to DIL surface"]

                        if dil_structure_info in target_dil_by_centroid_dict:
                            target_dil_by_centroid_bool = True
                        else:
                            target_dil_by_centroid_bool = False

                        if dil_structure_info in target_dil_by_surfaces_dict:
                            target_dil_by_surface_bool = True
                        else:
                            target_dil_by_surface_bool = False

                        patientUID_list.append(patientUID)
                        structureID_list.append(structureID)
                        sim_type_list.append(bx_simulated_type)
                        sim_bool_list.append(bx_simulated_bool)
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
                        x_dil_centroid_frame_list.append(x_dil_centroid_frame)
                        y_dil_centroid_frame_list.append(y_dil_centroid_frame)
                        z_dil_centroid_frame_list.append(z_dil_centroid_frame)
                        dist_cent_to_cent_list.append(dist_cent_to_cent)
                        nn_dist_surf_to_surf_list.append(nn_dist_surf_to_surf)
                        target_dil_by_centroids_list.append(target_dil_by_centroid_bool)
                        target_dil_by_surfaces_list.append(target_dil_by_surface_bool)
                        bx_position_in_prostate_LR_list.append(bx_LR_pos)
                        bx_position_in_prostate_AP_list.append(bx_AP_pos)
                        bx_position_in_prostate_SI_list.append(bx_SI_pos)
                        bx_relative_reference_prostate_structure_type_list.append(bx_relative_prostate_struct_type)
                        bx_relative_reference_prostate_id_list.append(bx_relative_prostate_id)
                        bx_relative_reference_prostate_index_list.append(bx_relative_prostate_index)
                        
                    else:
                        pass

                                                                
                    sp_bx_relative_dil_info_dict = {"Patient ID": patientUID_list,
                                                    "Bx ID": structureID_list,
                                                    "Simulated bool": sim_bool_list,
                                                    "Simulated type": sim_type_list,
                                                    "Struct type": bx_ref_list,
                                                    "Bx refnum": structure_reference_number_list,
                                                    "Bx index": specific_structure_index_list,
                                                    "Relative DIL ID": dil_structureID_list,
                                                    "Relative struct type": dil_ref_list,
                                                    "Relative DIL ref num": dil_structure_reference_number_list,
                                                    "Relative DIL index": specific_dil_structure_index_list,
                                                    "Target DIL (by centroids)": target_dil_by_centroids_list,
                                                    "Target DIL (by surfaces)": target_dil_by_surfaces_list,
                                                    "BX centroid vec": bx_centroid_vec_list,
                                                    "DIL centroid vec": dil_centroid_vec_list,
                                                    "BX to DIL centroid vector": vector_cent_to_cent_list,
                                                    "BX to DIL centroid (X)": x_cent_to_cent_list,
                                                    "BX to DIL centroid (Y)": y_cent_to_cent_list,
                                                    "BX to DIL centroid (Z)": z_cent_to_cent_list,
                                                    "Bx (X, DIL centroid frame)": x_dil_centroid_frame_list,
                                                    "Bx (Y, DIL centroid frame)": y_dil_centroid_frame_list,
                                                    "Bx (Z, DIL centroid frame)": z_dil_centroid_frame_list,
                                                    "BX to DIL centroid distance": dist_cent_to_cent_list,
                                                    "NN surface-surface distance": nn_dist_surf_to_surf_list,
                                                    "Relative prostate ID": bx_relative_reference_prostate_id_list,
                                                    "Relative prostate struct type": bx_relative_reference_prostate_structure_type_list,
                                                    "Relative prostate index": bx_relative_reference_prostate_index_list,
                                                    "Bx position in prostate LR": bx_position_in_prostate_LR_list,
                                                    "Bx position in prostate AP": bx_position_in_prostate_AP_list,
                                                    "Bx position in prostate SI": bx_position_in_prostate_SI_list
                                                    }
                    
                    sp_bx_relative_dil_dataframe = pandas.DataFrame.from_dict(data = sp_bx_relative_dil_info_dict)

                    sp_patient_relative_dil_dataframe_list.append(sp_bx_relative_dil_dataframe)

                    

        sp_patient_relative_dil_dataframe = pandas.concat(sp_patient_relative_dil_dataframe_list)
        
        sp_patient_relative_dil_dataframe = convert_columns_to_categorical_and_downcast(sp_patient_relative_dil_dataframe, threshold=0.25)

        cohort_nearest_dils_dataframe = pandas.concat([cohort_nearest_dils_dataframe,sp_patient_relative_dil_dataframe]).reset_index(drop=True)

        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Nearest DILs info dataframe"] = sp_patient_relative_dil_dataframe

    cohort_nearest_dils_dataframe = convert_columns_to_categorical_and_downcast(cohort_nearest_dils_dataframe, threshold=0.25)

    return cohort_nearest_dils_dataframe




def bx_info_dataframe_builder(cohort_nearest_dils_dataframe,
                            cohort_global_tissue_class_by_tissue_type_dataframe,
                            cohort_all_bx_dvh_metric_dataframe,
                            ):

    bx_info_cohort_dataframe = pandas.merge(cohort_nearest_dils_dataframe, cohort_global_tissue_class_by_tissue_type_dataframe, on=['Patient ID', 'Bx ID', 'Bx index', 'Bx refnum'], how ='outer', suffixes=('', '_drop'))

    bx_info_cohort_dataframe = bx_info_cohort_dataframe[[col for col in bx_info_cohort_dataframe.columns if not col.endswith('_drop')]]


    bx_info_cohort_dataframe = pandas.merge(bx_info_cohort_dataframe, cohort_all_bx_dvh_metric_dataframe, on=['Patient ID', 'Bx ID', 'Bx index'],  how ='outer', suffixes=('', '_drop'))
    
    bx_info_cohort_dataframe = bx_info_cohort_dataframe[[col for col in bx_info_cohort_dataframe.columns if not col.endswith('_drop')]]



    return bx_info_cohort_dataframe


def dil_optimization_results_dataframe_builder(master_structure_reference_dict,
                                       all_ref_key,
                                       dil_ref
                                       ):
     
     for patientUID,pydicom_item in master_structure_reference_dict.items():
        dil_centroids_optimization_locations_dataframe_list = []
        optimal_locations_dataframe_list = []
        potential_optimal_locations_dataframe_list = []
        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):

            dil_centroids_optimization_locations_dataframe = specific_dil_structure["Biopsy optimization: DIL centroid optimal biopsy location dataframe"]
            optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"]
            potential_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"]

            dil_centroids_optimization_locations_dataframe_list.append(dil_centroids_optimization_locations_dataframe)
            optimal_locations_dataframe_list.append(optimal_locations_dataframe)
            potential_optimal_locations_dataframe_list.append(potential_optimal_locations_dataframe)

        sp_patient_dil_centroids_optimization_dataframe = pandas.concat(dil_centroids_optimization_locations_dataframe_list, ignore_index = True)
        sp_patient_optimal_dataframe = pandas.concat(optimal_locations_dataframe_list, ignore_index = True)
        sp_patient_potential_optimal_dataframe = pandas.concat(potential_optimal_locations_dataframe_list, ignore_index = True)

        sp_patient_dil_centroids_optimization_dataframe = convert_columns_to_categorical_and_downcast(sp_patient_dil_centroids_optimization_dataframe, threshold=0.25)
        sp_patient_optimal_dataframe = convert_columns_to_categorical_and_downcast(sp_patient_optimal_dataframe, threshold=0.25)
        sp_patient_potential_optimal_dataframe = convert_columns_to_categorical_and_downcast(sp_patient_potential_optimal_dataframe, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - DIL centroids optimal targeting dataframe"] = sp_patient_dil_centroids_optimization_dataframe
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - Optimal DIL targeting dataframe"] = sp_patient_optimal_dataframe
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - Optimal DIL targeting entire lattice dataframe"] = sp_patient_potential_optimal_dataframe





def dvh_metrics_dataframe_builder_sp_biopsy(master_structure_reference_dict,
                                            bx_ref,
                                            all_ref_key):
    
    cohort_all_bx_dvh_metric_dataframe = pandas.DataFrame()
    for patientUID,pydicom_item in master_structure_reference_dict.items():
        all_bx_dvh_metrics_dataframe = pandas.DataFrame()
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):

            dvh_metric_dataframe_per_biopsy = specific_bx_structure["Output data frames"]["DVH metrics"]

            all_bx_dvh_metrics_dataframe = pandas.concat([all_bx_dvh_metrics_dataframe,dvh_metric_dataframe_per_biopsy], ignore_index = True)

            del specific_bx_structure["Output data frames"]["DVH metrics"]
        
        all_bx_dvh_metrics_dataframe = convert_columns_to_categorical_and_downcast(all_bx_dvh_metrics_dataframe, threshold=0.25)

        pydicom_item[all_ref_key]["Multi-structure MC simulation output dataframes dict"]["DVH metrics"] = all_bx_dvh_metrics_dataframe

        cohort_all_bx_dvh_metric_dataframe = pandas.concat([cohort_all_bx_dvh_metric_dataframe, all_bx_dvh_metrics_dataframe], ignore_index = True)
    
    cohort_all_bx_dvh_metric_dataframe = convert_columns_to_categorical_and_downcast(cohort_all_bx_dvh_metric_dataframe, threshold=0.25)

    return cohort_all_bx_dvh_metric_dataframe

def normal_CI_estimator_by_dataframe_row(row, mean_col_name = 'Mean', std_err_col_name = 'Std err'):
    row_ci= mf.normal_CI_estimator(row[mean_col_name], row[std_err_col_name])
    return row_ci









def bx_global_score_to_target_dil_3d_radiomic_features_dataframe_builder(structure_cohort_3d_radiomic_features_dataframe,
                                                                         cohort_global_tissue_class_by_structure_dataframe,
                                                                         master_structure_reference_dict,
                                                                         bx_ref
                                                                         ):
    

    cohort_global_tissue_scores_with_target_dil_radiomic_features_df = pandas.DataFrame()

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
            target_dict = specific_bx_structure["Target DIL by centroid dict"]

            bx_roi = specific_bx_structure["ROI"]
            bx_refnum = specific_bx_structure["Ref #"]
            
            target_tuple_info = list(target_dict.keys())[0]
            target_id = target_tuple_info[0]
            target_struct_type = target_tuple_info[1]
            target_refnum = target_tuple_info[2]
            target_index = target_tuple_info[3]

            
            sp_bx_global_scores_by_structure = cohort_global_tissue_class_by_structure_dataframe[(cohort_global_tissue_class_by_structure_dataframe['Patient ID'] == patientUID) &
                                                                (cohort_global_tissue_class_by_structure_dataframe['Bx ID'] == bx_roi) &
                                                               (cohort_global_tissue_class_by_structure_dataframe["Bx index"] == specific_bx_structure_index) &
                                                               (cohort_global_tissue_class_by_structure_dataframe["Bx refnum"] == bx_refnum) & 
                                                               (cohort_global_tissue_class_by_structure_dataframe["Relative structure ROI"] == target_id) &
                                                               (cohort_global_tissue_class_by_structure_dataframe["Relative structure type"] == target_struct_type) &
                                                               (cohort_global_tissue_class_by_structure_dataframe["Relative structure index"] == target_index)] 
            

            sp_target_structure_radiomic_features = structure_cohort_3d_radiomic_features_dataframe[(structure_cohort_3d_radiomic_features_dataframe['Patient ID'] == patientUID) &
                                                            (structure_cohort_3d_radiomic_features_dataframe['Structure ID'] == target_id) & 
                                                            (structure_cohort_3d_radiomic_features_dataframe['Structure type'] == target_struct_type) & 
                                                            (structure_cohort_3d_radiomic_features_dataframe['Structure refnum'] == target_refnum) ]
            

            sp_patient_sp_bx_global_tissue_scores_with_target_dil_radiomic_features_df = pandas.merge(sp_bx_global_scores_by_structure, 
                                                                                        sp_target_structure_radiomic_features, 
                                                                                        on='Patient ID', 
                                                                                        suffixes=('_scores_df', '_radiomics_df'))
            
            cohort_global_tissue_scores_with_target_dil_radiomic_features_df = pandas.concat([cohort_global_tissue_scores_with_target_dil_radiomic_features_df, sp_patient_sp_bx_global_tissue_scores_with_target_dil_radiomic_features_df], ignore_index = True)

    cohort_global_tissue_scores_with_target_dil_radiomic_features_df = convert_columns_to_categorical_and_downcast(cohort_global_tissue_scores_with_target_dil_radiomic_features_df, threshold=0.25)

    return cohort_global_tissue_scores_with_target_dil_radiomic_features_df












def concatenate_with_categoricals(df1, df2, cat_columns):
    """
    Concatenates two dataframes while ensuring specified columns remain categorical
    with unified categories if necessary.
    
    Parameters:
        df1 (pd.DataFrame): First DataFrame to concatenate.
        df2 (pd.DataFrame): Second DataFrame to concatenate.
        cat_columns (list of str): List of column names to treat as categorical.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame with specified columns as categorical.
    """
    # Ensure specified columns are categorical in both dataframes
    for column in cat_columns:
        if column in df1.columns and column in df2.columns:
            # Unify categories between the two DataFrames
            unified_categories = union_categoricals([df1[column], df2[column]])
            df1[column] = pandas.Categorical(df1[column], categories=unified_categories.categories)
            df2[column] = pandas.Categorical(df2[column], categories=unified_categories.categories)
        elif column in df1.columns:
            df1[column] = df1[column].astype('category')
        elif column in df2.columns:
            df2[column] = df2[column].astype('category')

    # Concatenate the dataframes
    result_df = pandas.concat([df1, df2], ignore_index=True)
    
    # Re-apply categorical type to ensure it's maintained post-concatenation
    for column in cat_columns:
        if column in result_df.columns:
            result_df[column] = pandas.Categorical(result_df[column], categories=result_df[column].cat.categories)

    return result_df


# def concatenate_with_auto_categoricals(df1, df2, threshold = 0.25, cat_columns=None):
#     """
#     Concatenates two dataframes while automatically deciding or ensuring specified columns
#     remain categorical with unified categories if necessary.
    
#     If cat_columns is not specified, the function checks each column to determine if 
#     converting it to categorical would be beneficial based on the unique values.
    
#     Parameters:
#         df1 (pd.DataFrame): First DataFrame to concatenate.
#         df2 (pd.DataFrame): Second DataFrame to concatenate.
#         cat_columns (list of str, optional): List of column names to treat as categorical.
    
#     Returns:
#         pd.DataFrame: Concatenated DataFrame with optimized column types.
#     """
#     if cat_columns is None:
#         # Automatically determine which columns could be beneficial to convert to categorical
#         cat_columns = []
#         for column in set(df1.columns).intersection(df2.columns):  # Consider only common columns
#             unique_vals = pandas.unique(pandas.concat([df1[column], df2[column]]))
#             if len(unique_vals) / (len(df1[column]) + len(df2[column])) < threshold:  # Arbitrary threshold
#                 cat_columns.append(column)

#     # Ensure specified or determined columns are categorical in both dataframes and unify categories
#     for column in cat_columns:
#         if column in df1.columns and column in df2.columns:
#             # Create a unified categorical series
#             unified_categories = union_categoricals([df1[column], df2[column]])
#             df1[column] = pandas.Categorical(df1[column], categories=unified_categories.categories)
#             df2[column] = pandas.Categorical(df2[column], categories=unified_categories.categories)
#         elif column in df1.columns:
#             df1[column] = df1[column].astype('category')
#         elif column in df2.columns:
#             df2[column] = df2[column].astype('category')

#     # Concatenate the dataframes
#     return pandas.concat([df1, df2], ignore_index=True)


# def concatenate_with_auto_categoricals(df1, df2, threshold=0.25, cat_columns=None):
#     """
#     Concatenates two dataframes while automatically deciding or ensuring specified columns
#     remain categorical with unified categories if necessary.
    
#     If cat_columns is not specified, the function checks each column to determine if 
#     converting it to categorical would be beneficial based on the unique values.
    
#     Parameters:
#         df1 (pd.DataFrame): First DataFrame to concatenate.
#         df2 (pd.DataFrame): Second DataFrame to concatenate.
#         cat_columns (list of str, optional): List of column names to treat as categorical.
#         threshold (float): Ratio of unique values to total values which justifies conversion to categorical.
    
#     Returns:
#         pd.DataFrame: Concatenated DataFrame with optimized column types.
#     """
#     if cat_columns is None:
#         # Automatically determine which columns could be beneficial to convert to categorical
#         cat_columns = []
#         for column in set(df1.columns).intersection(df2.columns):  # Consider only common columns
#             combined_column = pandas.concat([df1[column], df2[column]])
#             unique_vals = pandas.unique(combined_column)
#             if len(unique_vals) / len(combined_column) < threshold:
#                 cat_columns.append(column)

#     # Ensure specified or determined columns are categorical in both dataframes and unify categories
#     for column in cat_columns:
#         if column in df1.columns and column in df2.columns:
#             # Determine the data type that should be used for the categorical type
#             common_dtype = pandas.api.types.find_common_type([df1[column].dtype, df2[column].dtype])

#             # Create a unified categorical series
#             unified_categories = union_categoricals([pandas.Categorical(df1[column], dtype=common_dtype),
#                                                      pandas.Categorical(df2[column], dtype=common_dtype)])

#             df1[column] = pandas.Categorical(df1[column], categories=unified_categories.categories, dtype=common_dtype)
#             df2[column] = pandas.Categorical(df2[column], categories=unified_categories.categories, dtype=common_dtype)
#         elif column in df1.columns:
#             df1[column] = pandas.Categorical(df1[column])
#         elif column in df2.columns:
#             df2[column] = pandas.Categorical(df2[column])

#     # Concatenate the dataframes
#     return pandas.concat([df1, df2], ignore_index=True)



def concatenate_with_auto_categoricals(df1, df2, threshold=0.25, cat_columns=None):
    """
    Concatenates two dataframes while automatically deciding or ensuring specified columns
    remain categorical with unified categories if necessary.
    
    If cat_columns is not specified, the function checks each column to determine if 
    converting it to categorical would be beneficial based on the unique values. Data types
    are only converted to string if they differ between the two DataFrames.
    
    Parameters:
        df1 (pd.DataFrame): First DataFrame to concatenate.
        df2 (pd.DataFrame): Second DataFrame to concatenate.
        cat_columns (list of str, optional): List of column names to treat as categorical.
        threshold (float): Ratio of unique values to total values which justifies conversion to categorical.
    
    Returns:
        pd.DataFrame: Concatenated DataFrame with optimized column types.
    """
    if cat_columns is None:
        # Automatically determine which columns could be beneficial to convert to categorical
        cat_columns = []
        for column in set(df1.columns).intersection(df2.columns):  # Consider only common columns
            combined_column = pandas.concat([df1[column], df2[column]])
            unique_vals = pandas.unique(combined_column)
            if len(unique_vals) / len(combined_column) < threshold:
                cat_columns.append(column)

    # Ensure specified or determined columns are categorical in both dataframes and unify categories
    for column in cat_columns:
        if column in df1.columns and column in df2.columns:
            # Check if data types differ
            dtype1 = df1[column].dtype
            dtype2 = df2[column].dtype
            
            if dtype1 != dtype2 or not (pandas.api.types.is_categorical_dtype(dtype1) and pandas.api.types.is_categorical_dtype(dtype2)):
                # Convert data types to string if they differ or are not categorical
                df1[column] = df1[column].astype(str)
                df2[column] = df2[column].astype(str)
            
            # Convert to categorical and unify categories
            unified_categories = union_categoricals([df1[column].astype('category'), df2[column].astype('category')])
            df1[column] = pandas.Categorical(df1[column], categories=unified_categories.categories)
            df2[column] = pandas.Categorical(df2[column], categories=unified_categories.categories)
        elif column in df1.columns:
            df1[column] = df1[column].astype('category')
        elif column in df2.columns:
            df2[column] = df2[column].astype('category')

    # Concatenate the dataframes
    return pandas.concat([df1, df2], ignore_index=True)

def convert_columns_to_categorical_v2(df, threshold=0.25):
    """
    Converts DataFrame columns to categorical types based on a uniqueness threshold.
    
    Parameters:
        df (pd.DataFrame): DataFrame whose columns are to be examined and potentially converted.
        threshold (float): Maximum ratio of unique values to total entries that allows conversion to categorical.
                           Default is 0.25 (25%).
    
    Returns:
        pd.DataFrame: DataFrame with columns converted to categorical where applicable.
    """
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Calculate the number of unique values
        num_unique_values = df[column].nunique()
        # Calculate the total number of entries in the column
        total_entries = len(df[column])
        # Calculate the ratio of unique values to total entries
        unique_ratio = num_unique_values / total_entries
        
        # Convert column to categorical if the ratio of unique values is below the threshold
        if unique_ratio <= threshold:
            df[column] = pandas.Categorical(df[column])
    
    return df



def convert_columns_to_categorical_v3(df, threshold=0.25, ignore_types=(np.floating,)):
    """
    Converts DataFrame columns to categorical types based on a uniqueness threshold,
    ignoring columns of specified data types, including all subtypes of each data type.

    Parameters:
        df (pd.DataFrame): DataFrame whose columns are to be examined and potentially converted.
        threshold (float): Maximum ratio of unique values to total entries that allows conversion to categorical.
                           Default is 0.25 (25%).
        ignore_types (tuple): Tuple of data types to ignore. Defaults to (np.floating,) which covers all float types.

    Returns:
        pd.DataFrame: DataFrame with columns converted to categorical where applicable.
    """
    for column in df.columns:
        # Skip conversion if the column is already categorical
        if pandas.api.types.is_categorical_dtype(df[column].dtype):
            continue

        # Ensure that the dtype is a numpy dtype before checking subtypes with np.issubdtype
        column_dtype = df[column].dtype
        if hasattr(column_dtype, 'type') and any(np.issubdtype(column_dtype.type, t) for t in ignore_types):
            continue

        # Calculate the number of unique values and the total number of entries in the column
        num_unique_values = df[column].nunique()
        total_entries = len(df[column])

        # Calculate the ratio of unique values to total entries
        unique_ratio = num_unique_values / total_entries
        
        # Convert column to categorical if the ratio of unique values is below the threshold
        if unique_ratio <= threshold:
            df[column] = pandas.Categorical(df[column])
    
    return df


def convert_columns_to_categorical_and_downcast(df, threshold=0.25, ignore_types=(np.floating,)):
    """
    Converts DataFrame columns to categorical types based on a uniqueness threshold,
    ignoring columns of specified data types, including all subtypes of each data type.
    Numeric columns are only downcasted if they are not converted to categorical.

    Parameters:
        df (pd.DataFrame): DataFrame whose columns are to be examined and potentially converted.
        threshold (float): Maximum ratio of unique values to total entries that allows conversion to categorical.
                           Default is 0.25 (25%).
        ignore_types (tuple): Tuple of data types to ignore. Defaults to (np.floating,) which covers all float types.

    Returns:
        pd.DataFrame: DataFrame with columns converted to categorical where applicable.
    """
    for column in df.columns:
        if pandas.api.types.is_categorical_dtype(df[column].dtype):
            continue  # Skip if already categorical

        # Calculate the number of unique values and the total number of entries in the column
        num_unique_values = df[column].nunique()
        total_entries = len(df[column])
        unique_ratio = num_unique_values / total_entries

        # Convert column to categorical if the ratio of unique values is below the threshold
        if unique_ratio <= threshold and not any(np.issubdtype(df[column].dtype, t) for t in ignore_types):
            df[column] = pandas.Categorical(df[column])
        elif pandas.api.types.is_numeric_dtype(df[column]):
            # Downcast numeric columns that were not converted to categorical
            if pandas.api.types.is_integer_dtype(df[column]):
                df[column] = pandas.to_numeric(df[column], downcast='integer')
            elif pandas.api.types.is_float_dtype(df[column]):
                df[column] = pandas.to_numeric(df[column], downcast='float')

    return df
