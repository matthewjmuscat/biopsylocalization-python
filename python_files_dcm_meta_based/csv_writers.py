import numpy as np
import csv
import math_funcs as mf


def csv_writer_containment(live_display,
                           layout_groups,
                           master_structure_reference_dict,
                           master_structure_info_dict,
                           patient_sp_output_csv_dir_dict,
                           bx_ref,
                           cancer_tissue_label,
                           structure_miss_probability_roi,
                           miss_structure_complement_label
                           ):

    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, completed_sections_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list

    with live_display:
        live_display.start(refresh = True)

        num_MC_containment_simulations_input = master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"]

        patientUID_default = "Initializing"
        processing_patient_csv_writing_description = "Writing containment CSVs to file [{}]...".format(patientUID_default)
        processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_description, total = master_structure_info_dict["Global"]["Num cases"])
        processing_patient_csv_writing_description_completed = "Writing containment CSVs to file"
        processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)
        
        for patientUID,pydicom_item in master_structure_reference_dict.items():

            processing_patient_csv_writing_description = "Writing containment CSVs to file [{}]...".format(patientUID)
            patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_description)

            patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                simulated_bool = specific_bx_structure["Simulated bool"]
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                bx_points_bx_coords_sys_arr_row = bx_points_bx_coords_sys_arr_list.copy()
                bx_points_bx_coords_sys_arr_row.insert(0,'Sampled point vector (Bx coord sys) (mm)')
                containment_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_c='+str(num_MC_containment_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_out.csv'
                containment_output_csv_file_path = patient_sp_output_csv_dir.joinpath(containment_output_file_name)
                with open(containment_output_csv_file_path, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(['Patient ID',patientUID])
                    write.writerow(['BX ID',specific_bx_structure['ROI']])
                    write.writerow(['Simulated', simulated_bool])
                    write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                    write.writerow(['Num MC containment sims',num_MC_containment_simulations_input])
                    write.writerow(['Num bx pt samples',num_sample_pts_per_bx])
                    
                    # global tissue class
                    write.writerow(['---'])
                    write.writerow(['Global by class'])
                    write.writerow(['---'])
                    rows_to_write_list = []

                    containment_output_by_MC_trial_pandas_data_frame = specific_bx_structure["Output data frames"]["Mutual containment output by bx point"]
                    tissue_classes_list = [cancer_tissue_label,structure_miss_probability_roi,miss_structure_complement_label]
                    for tissue_class in tissue_classes_list:
                        tissue_class_row = ['Tissue type', tissue_class]

                        mean_prob = containment_output_by_MC_trial_pandas_data_frame[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == tissue_class]["Mean probability (binom est)"].mean()
                        mean_prob_row = ['Mean probability', mean_prob]

                        mean_std = containment_output_by_MC_trial_pandas_data_frame[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == tissue_class]["Mean probability (binom est)"].std()
                        mean_std_row = ['STD', mean_std]

                        mean_stderr = containment_output_by_MC_trial_pandas_data_frame[containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == tissue_class]["STD err"].mean()
                        std_err_row = ['STD err', mean_stderr]
                    
                        tissue_class_CI_tuple = mf.normal_CI_estimator(mean_prob, mean_stderr)
                        tissue_class_CI_lower_row = ['CI lower', tissue_class_CI_tuple[0]]
                        tissue_class_CI_upper_row = ['CI upper', tissue_class_CI_tuple[1]]

                        rows_to_write_list.append(['+++'])
                        rows_to_write_list.append(tissue_class_row)
                        rows_to_write_list.append(mean_prob_row)
                        rows_to_write_list.append(mean_std_row)
                        rows_to_write_list.append(std_err_row)
                        rows_to_write_list.append(tissue_class_CI_lower_row)
                        rows_to_write_list.append(tissue_class_CI_upper_row)
                        

                    for row_to_write in rows_to_write_list:
                        write.writerow(row_to_write)
                    
                    del rows_to_write_list

                    # global
                    write.writerow(['---'])
                    write.writerow(['Global by structure'])
                    write.writerow(['---'])
                    rows_to_write_list = []
                    for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                        containment_structure_ROI = containment_structure_key_tuple[0]
                        containment_structure_nominal_list = containment_structure_dict['Nominal containment list']
                        containment_structure_nominal_num_contained = sum(containment_structure_nominal_list)
                        containment_structure_nominal_percent_contained = (containment_structure_nominal_num_contained/len(containment_structure_nominal_list))*100
                        containment_structure_nominal_percent_contained_with_cont_anat_ROI_row = [containment_structure_ROI + ' Nominal percent volume contained',containment_structure_nominal_percent_contained]
                        rows_to_write_list.append(containment_structure_nominal_percent_contained_with_cont_anat_ROI_row)
                        

                        containment_structure_binom_est_arr = np.array(containment_structure_dict["Binomial estimator list"])
                        containment_structure_binom_est_global_mean = np.mean(containment_structure_binom_est_arr)
                        containment_structure_binom_est_std = np.std(containment_structure_binom_est_arr)
                        containment_structure_binom_est_std_err = containment_structure_binom_est_std / np.sqrt(np.size(containment_structure_binom_est_arr))
                        containment_structure_binom_est_CI = mf.normal_CI_estimator(containment_structure_binom_est_global_mean, containment_structure_binom_est_std_err)
                        containment_structure_binom_est_global_mean_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean probability', containment_structure_binom_est_global_mean]
                        containment_structure_binom_est_global_std_with_cont_anat_ROI_row = [containment_structure_ROI + ' STD', containment_structure_binom_est_std]
                        containment_structure_binom_est_global_std_err_with_cont_anat_ROI_row = [containment_structure_ROI + ' STD err in mean', containment_structure_binom_est_std_err]
                        containment_structure_binom_est_global_mean_CI_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean CI', containment_structure_binom_est_CI]
                        
                        rows_to_write_list.append(containment_structure_binom_est_global_mean_with_cont_anat_ROI_row)
                        rows_to_write_list.append(containment_structure_binom_est_global_std_with_cont_anat_ROI_row)
                        rows_to_write_list.append(containment_structure_binom_est_global_std_err_with_cont_anat_ROI_row)
                        rows_to_write_list.append(containment_structure_binom_est_global_mean_CI_with_cont_anat_ROI_row)
                        

                    for row_to_write in rows_to_write_list:
                        write.writerow(row_to_write)
                    
                    del rows_to_write_list

                    
                    # Point wise
                    write.writerow(['---'])
                    write.writerow(['Point-wise'])
                    write.writerow(['---'])
                    
                    write.writerow(['Row ->','Fixed containment structure'])
                    write.writerow(['Col ->','Fixed bx point'])
                    write.writerow(bx_points_bx_coords_sys_arr_row)
                    x_vals_row = [point_vec[0] for point_vec in bx_points_bx_coords_sys_arr_list]
                    x_vals_row.insert(0,'X coord (mm)')
                    y_vals_row = [point_vec[1] for point_vec in bx_points_bx_coords_sys_arr_list]
                    y_vals_row.insert(0,'Y coord (mm)')
                    z_vals_row = [point_vec[2] for point_vec in bx_points_bx_coords_sys_arr_list]
                    z_vals_row.insert(0,'Z coord (mm)')
                    pt_radius_bx_coord_sys_row = [np.linalg.norm(point_vec[0:2]) for point_vec in bx_points_bx_coords_sys_arr_list]
                    pt_radius_bx_coord_sys_row.insert(0,'Cyl coord radius (mm)')
                    write.writerow(x_vals_row)
                    write.writerow(y_vals_row)
                    write.writerow(z_vals_row)
                    write.writerow(pt_radius_bx_coord_sys_row)
                    
                    rows_to_write_list = []
                    for containment_structure_key_tuple, containment_structure_dict in specific_bx_structure['MC data: compiled sim results'].items():
                        containment_structure_ROI = containment_structure_key_tuple[0]

                        containment_structure_nominal_list = containment_structure_dict['Nominal containment list']
                        containment_structure_nominal_with_cont_anat_ROI_row = [containment_structure_ROI + ' Nominal containment (0 or 1)']+containment_structure_nominal_list
                        rows_to_write_list.append(containment_structure_nominal_with_cont_anat_ROI_row)
                        
                        containment_structure_successes_list = containment_structure_dict['Total successes (containment) list']
                        containment_structure_successes_with_cont_anat_ROI_row = [containment_structure_ROI + ' Total successes']+containment_structure_successes_list
                        rows_to_write_list.append(containment_structure_successes_with_cont_anat_ROI_row)

                        containment_structure_binom_est_list = containment_structure_dict["Binomial estimator list"]
                        containment_structure_binom_est_with_cont_anat_ROI_row = [containment_structure_ROI + ' Mean probability']+containment_structure_binom_est_list
                        rows_to_write_list.append(containment_structure_binom_est_with_cont_anat_ROI_row)

                        containment_structure_stand_err_list = containment_structure_dict["Standard error (containment) list"]
                        containment_structure_stand_err_with_cont_anat_ROI_row = [containment_structure_ROI + ' SE']+containment_structure_stand_err_list
                        rows_to_write_list.append(containment_structure_stand_err_with_cont_anat_ROI_row)

                        containment_structure_conf_int_arr = np.array(containment_structure_dict["Confidence interval 95 (containment) list"]).T
                        containment_structure_conf_int_lower_with_cont_anat_ROI_row = [containment_structure_ROI + ' 95% CI lower'] + containment_structure_conf_int_arr[0].tolist()
                        rows_to_write_list.append(containment_structure_conf_int_lower_with_cont_anat_ROI_row)
                        containment_structure_conf_int_upper_with_cont_anat_ROI_row = [containment_structure_ROI + ' 95% CI upper'] + containment_structure_conf_int_arr[1].tolist()
                        rows_to_write_list.append(containment_structure_conf_int_upper_with_cont_anat_ROI_row)



                    # tumor tissue probabilities
                    tumor_containment_structure_nominal_list = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue nominal arr"].tolist()
                    tumor_containment_structure_binom_est_list = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue binomial est arr"].tolist()
                    tumor_containment_structure_std_err_list = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue standard error arr"].tolist()
                    tumor_containment_structure_CI_arr = specific_bx_structure["MC data: tumor tissue probability"]["Tumor tissue confidence interval 95 arr"]
                    tumor_containment_structure_CI_lower_list = tumor_containment_structure_CI_arr[0].tolist()
                    tumor_containment_structure_CI_upper_list = tumor_containment_structure_CI_arr[1].tolist()
                    
                    tumor_containment_structure_nominal_row = ["Tumor nominal containment (0 or 1)"] + tumor_containment_structure_nominal_list
                    tumor_containment_structure_binom_row = ["Tumor Mean probability"] + tumor_containment_structure_binom_est_list
                    tumor_containment_structure_STD_row = ["Tumor SE"] + tumor_containment_structure_std_err_list
                    tumor_containment_structure_CI_lower_row = ["Tumor 95% CI lower"] + tumor_containment_structure_CI_lower_list
                    tumor_containment_structure_CI_upper_row = ["Tumor 95% CI upper"] + tumor_containment_structure_CI_upper_list

                    rows_to_write_list.append(tumor_containment_structure_nominal_row)
                    rows_to_write_list.append(tumor_containment_structure_binom_row)
                    rows_to_write_list.append(tumor_containment_structure_STD_row)
                    rows_to_write_list.append(tumor_containment_structure_CI_lower_row)
                    rows_to_write_list.append(tumor_containment_structure_CI_upper_row)

                    # miss structure probabilities
                    miss_structure_roi = specific_bx_structure["MC data: miss structure tissue probability"]['OAR miss structure info'][0]
                    miss_structure_nominal_list = specific_bx_structure["MC data: miss structure tissue probability"]["OAR tissue miss nominal arr"].tolist()
                    miss_structure_binom_est_list = specific_bx_structure["MC data: miss structure tissue probability"]["OAR tissue miss binomial est arr"].tolist()
                    miss_structure_std_err_list = specific_bx_structure["MC data: miss structure tissue probability"]["OAR tissue standard error arr"].tolist()
                    miss_structure_CI_arr = specific_bx_structure["MC data: miss structure tissue probability"]["OAR tissue miss confidence interval 95 2d arr"]
                    miss_structure_CI_lower_list = miss_structure_CI_arr[0].tolist()
                    miss_structure_CI_upper_list = miss_structure_CI_arr[1].tolist()
                    
                    miss_structure_ROI_row = ["Miss structure"]+[miss_structure_roi]
                    miss_structure_nominal_row = ["OAR miss nominal containment (0 or 1)"] + miss_structure_nominal_list
                    miss_structure_binom_row = ["OAR miss Mean probability"] + miss_structure_binom_est_list
                    miss_structure_STD_row = ["OAR miss SE"] + miss_structure_std_err_list
                    miss_structure_CI_lower_row = ["OAR miss 95% CI lower"] + miss_structure_CI_lower_list
                    miss_structure_CI_upper_row = ["OAR miss 95% CI upper"] + miss_structure_CI_upper_list

                    rows_to_write_list.append(miss_structure_ROI_row)
                    rows_to_write_list.append(miss_structure_nominal_row)
                    rows_to_write_list.append(miss_structure_binom_row)
                    rows_to_write_list.append(miss_structure_STD_row)
                    rows_to_write_list.append(miss_structure_CI_lower_row)
                    rows_to_write_list.append(miss_structure_CI_upper_row)
                    
                    """
                    # tumor length estimate 
                    rows_to_write_list.append(['---'])
                    rows_to_write_list.append(['Tumor length estimate'])
                    rows_to_write_list.append(['---'])

                    
                    tissue_length_by_threshold_dict = specific_bx_structure["MC data: tissue length above threshold dict"] 
                    for key,item in tissue_length_by_threshold_dict.items():
                        probability_threshold = key
                        length_est_dist_list = item["Length estimate distribution"].tolist()
                        num_bootstraps = item["Num bootstraps"]
                        length_estimate_mean = item["Length estimate mean"]
                        leangth_estimate_se = item["Length estimate se"]

                        length_estimate_probability_threshold_row = ["Probability threshold"] + [probability_threshold]
                        length_estimate_distribution_row = ["Length estimate bootstrap distribution"] + length_est_dist_list
                        length_estimate_num_bootstraps_row = ["Num bootstraps"]+ [num_bootstraps]
                        length_estimate_mean_row = ["Length estimate mean"] + [length_estimate_mean]
                        length_estimate_se_row = ["Length estimate se"] + [leangth_estimate_se]

                        rows_to_write_list.append(['+++'])
                        rows_to_write_list.append(length_estimate_probability_threshold_row)
                        rows_to_write_list.append(length_estimate_distribution_row)
                        rows_to_write_list.append(length_estimate_num_bootstraps_row)
                        rows_to_write_list.append(length_estimate_mean_row)
                        rows_to_write_list.append(length_estimate_se_row)
                        
                    """

                    rows_to_write_list.append(['---'])
                    
                    for row_to_write in rows_to_write_list:
                        write.writerow(row_to_write)
                    
                    del rows_to_write_list



                        
            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_completed_task, advance = 1)

        patients_progress.update(processing_patients_task, visible = False)
        completed_progress.update(processing_patients_completed_task, visible = True)

                        


        patientUID_default = "Initializing"
        processing_patient_csv_writing_voxelized_description = "Writing containment CSVs (voxelized) to file [{}]...".format(patientUID_default)
        processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_voxelized_description, total = master_structure_info_dict["Global"]["Num cases"])
        processing_patient_csv_writing_voxelized_description_completed = "Writing containment CSVs (voxelized) to file"
        processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_voxelized_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)
        
        for patientUID,pydicom_item in master_structure_reference_dict.items():

            processing_patient_csv_writing_voxelized_description = "Writing containment CSVs (voxelized) to file [{}]...".format(patientUID)
            patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_voxelized_description)

            patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                containment_voxelized_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_c='+str(num_MC_containment_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-containment_voxelized_out.csv'
                containment_voxelized_output_csv_file_path = patient_sp_output_csv_dir.joinpath(containment_voxelized_output_file_name)
                with open(containment_voxelized_output_csv_file_path, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(['Patient ID ->',patientUID])
                    write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                    write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                    write.writerow(['Num MC containment sims ->',num_MC_containment_simulations_input])
                    write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                    write.writerow(['Row ->','Fixed containment structure'])
                    write.writerow(['Col ->','Fixed voxel'])
                                                
                    for containment_structure_key_tuple, voxelized_containment_structure_dict in specific_bx_structure["MC data: voxelized containment results dict (dict of lists)"].items():
                        containment_structure_ROI = containment_structure_key_tuple[0]
                        num_voxels = voxelized_containment_structure_dict["Num voxels"]
                        voxel_index_row = list(range(num_voxels))
                        voxel_index_row.insert(0,'Voxel index')
                        biopsy_z_voxel_range_row = voxelized_containment_structure_dict["Voxel z range"].copy()
                        rounded_biopsy_z_voxel_range_row = [[round(sub_list[0],2),round(sub_list[1],2)] for sub_list in biopsy_z_voxel_range_row]
                        biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                        rounded_biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                        num_sample_pts_in_voxel_row = voxelized_containment_structure_dict["Num sample pts in voxel"].copy()
                        num_sample_pts_in_voxel_row.insert(0, 'Num sample pts in vxl')
                        arth_mean_binomial_estimator_row = voxelized_containment_structure_dict["Arithmetic mean of binomial estimators in voxel"].copy()
                        arth_mean_binomial_estimator_row.insert(0, 'Arth mean (binomial estimator)')
                        std_dev_binomial_estimator_row = voxelized_containment_structure_dict["Std dev of binomial estimators in voxel"].copy()
                        std_dev_binomial_estimator_row.insert(0, 'Std dev (binomial estimator)')

                        write.writerow([containment_structure_ROI])
                        write.writerow(voxel_index_row)
                        write.writerow(biopsy_z_voxel_range_row)
                        write.writerow(rounded_biopsy_z_voxel_range_row)
                        write.writerow(num_sample_pts_in_voxel_row)
                        write.writerow(arth_mean_binomial_estimator_row)
                        write.writerow(std_dev_binomial_estimator_row)
                        write.writerow([''])        


            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_completed_task, advance = 1)

        patients_progress.update(processing_patients_task, visible = False)
        completed_progress.update(processing_patients_completed_task, visible = True)
    



def csv_writer_dosimetry(live_display,
                           layout_groups,
                           master_structure_reference_dict,
                           master_structure_info_dict,
                           patient_sp_output_csv_dir_dict,
                           bx_ref,
                           display_dvh_as,
                           v_percent_DVH_to_calc_list
                           ):

    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, completed_sections_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list

    with live_display:
        live_display.start(refresh = True)

        num_MC_dose_simulations_input = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
    

        patientUID_default = "Initializing"
        processing_patient_csv_writing_description = "Writing dosimetry CSVs to file [{}]...".format(patientUID_default)
        processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_description, total = master_structure_info_dict["Global"]["Num cases"])
        processing_patient_csv_writing_description_completed = "Writing dosimetry CSVs to file"
        processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)
        
        for patientUID,pydicom_item in master_structure_reference_dict.items():

            processing_patient_csv_writing_description = "Writing dosimetry CSVs to file [{}]...".format(patientUID)
            patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_description)

            patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                simulated_bool = specific_bx_structure["Simulated bool"]
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                bx_points_bx_coords_sys_arr_list = list(bx_points_bx_coords_sys_arr)
                
                differential_dvh_dict = specific_bx_structure["MC data: Differential DVH dict"]
                cumulative_dvh_dict = specific_bx_structure["MC data: Cumulative DVH dict"]

                dvh_metric_vol_dose_percent_dict = specific_bx_structure["MC data: dose volume metrics dict"]
                
                dose_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_d='+str(num_MC_dose_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-dose_out.csv'
                dose_output_csv_file_path = patient_sp_output_csv_dir.joinpath(dose_output_file_name)
                specific_bx_structure["Output csv file paths dict"]["Dose output point-wise csv"] = dose_output_csv_file_path
                
                with open(dose_output_csv_file_path, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(['Patient ID ->',patientUID])
                    write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                    write.writerow(['Simulated', simulated_bool])
                    write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                    write.writerow(['Num MC dose sims ->',num_MC_dose_simulations_input])
                    write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                    
                    

                    stats_dose_val_all_MC_trials_by_bx_pt_list = specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"]
                    # global
                    
                    write.writerow(['---'])
                    write.writerow(['Global'])
                    write.writerow(['---'])
                    nominal_dose_by_bx_pt_arr = specific_bx_structure['MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)'][:,0]
                    nominal_mean_dose = np.mean(nominal_dose_by_bx_pt_arr)
                    nominal_std_dose = np.std(nominal_dose_by_bx_pt_arr)
                    nominal_std_err_dose = nominal_std_dose / np.sqrt(np.size(nominal_dose_by_bx_pt_arr))
                    nominal_dose_mean_CI = mf.normal_CI_estimator(nominal_mean_dose, nominal_std_err_dose)
                    
                    global_dose_by_bx_pt_arr = np.array(stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"])
                    global_mean_dose = np.mean(global_dose_by_bx_pt_arr)
                    global_std_dose = np.std(global_dose_by_bx_pt_arr)
                    global_std_err_dose = global_std_dose / np.sqrt(np.size(global_dose_by_bx_pt_arr))
                    global_dose_mean_CI = mf.normal_CI_estimator(global_mean_dose, global_std_err_dose)
                    quantile_95_arr = np.array(stats_dose_val_all_MC_trials_by_bx_pt_list["Quantiles dose by bx pt dict"]['Q95'])
                    quantile_5_arr = np.array(stats_dose_val_all_MC_trials_by_bx_pt_list["Quantiles dose by bx pt dict"]['Q5'])
                    quantile_95_5_difference_arr = quantile_95_arr - quantile_5_arr
                    quantile_95_5_difference_mean = np.mean(quantile_95_5_difference_arr)

                    quantile_95_mean_difference_arr = quantile_95_arr - global_dose_by_bx_pt_arr
                    quantile_95_mean_difference_mean = np.mean(quantile_95_mean_difference_arr)
                    mean_quantile_5_difference_arr = global_dose_by_bx_pt_arr - quantile_5_arr
                    mean_quantile_5_difference_mean = np.mean(mean_quantile_5_difference_arr)
                    
                    
                    write.writerow(['Nominal mean dose',nominal_mean_dose])
                    write.writerow(['Nominal std dose',nominal_std_dose])
                    write.writerow(['Nominal std err dose',nominal_std_err_dose])
                    write.writerow(['Nominal mean CI dose lower',nominal_dose_mean_CI[0]])
                    write.writerow(['Nominal mean CI dose upper',nominal_dose_mean_CI[1]])

                    write.writerow(['Global mean dose',global_mean_dose])
                    write.writerow(['Global std dose',global_std_dose])
                    write.writerow(['Global std err dose ',global_std_err_dose])
                    write.writerow(['Global mean CI dose lower',global_dose_mean_CI[0]])
                    write.writerow(['Global mean CI dose upper',global_dose_mean_CI[1]])
                    write.writerow(['Global mean (Q95-Q5)', quantile_95_5_difference_mean])
                    write.writerow(['Global mean (Q95-mean)', quantile_95_mean_difference_mean])
                    write.writerow(['Global mean (mean-Q5)', mean_quantile_5_difference_mean])



                    
                    
                    
                    # point-wise

                    write.writerow(['---'])
                    write.writerow(['Point-wise'])
                    write.writerow(['---'])
                    
                    write.writerow(['Row ->','Fixed bx pt'])
                    write.writerow(['Col ->','Fixed MC trial'])
                    write.writerow(['Vector (mm)',
                                    'X (mm)', 
                                    'Y (mm)', 
                                    'Z (mm)', 
                                    'r (mm)',
                                    'Nominal (Gy)', 
                                    'Mean (Gy)', 
                                    'STD (Gy)', 
                                    'All MC trials doses (Gy) -->'
                                    ])
                    
                    
                    for pt_index in range(num_sample_pts_per_bx):
                        #dose_vals_row_with_point = dose_vals_row.copy()
                        pt_radius_bx_coord_sys = np.linalg.norm(bx_points_bx_coords_sys_arr_list[pt_index][0:2])
                        mean_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["Mean dose by bx pt"][pt_index]
                        std_dose_val_specific_bx_pt = stats_dose_val_all_MC_trials_by_bx_pt_list["STD by bx pt"][pt_index]
                        nominal_dose_val_specific_bx_pt = specific_bx_structure['MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)'][:,0][pt_index]
                        info_row_part = [bx_points_bx_coords_sys_arr_list[pt_index], 
                                        bx_points_bx_coords_sys_arr_list[pt_index][0], 
                                        bx_points_bx_coords_sys_arr_list[pt_index][1], 
                                        bx_points_bx_coords_sys_arr_list[pt_index][2], 
                                        pt_radius_bx_coord_sys, 
                                        nominal_dose_val_specific_bx_pt,
                                        mean_dose_val_specific_bx_pt, 
                                        std_dose_val_specific_bx_pt
                                        ]
                        dose_vals_row_arr = specific_bx_structure['MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)'][:,1:][pt_index]
                        dose_vals_row_list = dose_vals_row_arr.tolist()
                        complete_dose_vals_row = info_row_part + dose_vals_row_list
                        write.writerow(complete_dose_vals_row)


                    for dvh_display_as_str in display_dvh_as:
                        if dvh_display_as_str == 'counts':
                            differential_dvh_histogram_counts_by_MC_trial_arr = differential_dvh_dict["Counts arr"]
                            cumulative_dvh_counts_by_MC_trial_arr = cumulative_dvh_dict["Counts arr"]
                        elif dvh_display_as_str == 'percent':
                            differential_dvh_histogram_counts_by_MC_trial_arr = differential_dvh_dict["Percent arr"]
                            cumulative_dvh_counts_by_MC_trial_arr = cumulative_dvh_dict["Percent arr"]
                        elif dvh_display_as_str == 'volume':
                            differential_dvh_histogram_counts_by_MC_trial_arr = differential_dvh_dict["Volume arr (cubic mm)"]
                            cumulative_dvh_counts_by_MC_trial_arr = cumulative_dvh_dict["Volume arr (cubic mm)"]
                        else:
                            continue
                        
                        differential_dvh_dose_bin_edges_1darr = differential_dvh_dict["Dose bins (edges) arr (Gy)"][0]
                                                    
                        write.writerow(['___'])
                        write.writerow(['Differential DVH info '+dvh_display_as_str])
                        write.writerow(['Each row is a fixed MC trial'])
                        write.writerow(['Lower dose bin edge (across)']+differential_dvh_dose_bin_edges_1darr.tolist()[0:-1])
                        write.writerow(['Upper dose bin edge (across)']+differential_dvh_dose_bin_edges_1darr.tolist()[1:])
                        write.writerow(['Trial number (down)'])
                        for mc_trial in range(differential_dvh_histogram_counts_by_MC_trial_arr.shape[0]):
                            if mc_trial == 0:
                                mc_trial_desc = 'Nominal'
                            else:
                                mc_trial_desc = str(mc_trial)
                            write.writerow([mc_trial_desc]+differential_dvh_histogram_counts_by_MC_trial_arr[mc_trial,:].tolist())


                        cumulative_dvh_dose_vals_by_MC_trial_1darr = cumulative_dvh_dict["Dose vals arr (Gy)"]
                        
                        write.writerow(['___'])
                        write.writerow(['Cumulative DVH info '+dvh_display_as_str])
                        write.writerow(['Each row is a fixed MC trial'])
                        write.writerow(['Dose value (across)']+cumulative_dvh_dose_vals_by_MC_trial_1darr.tolist())
                        write.writerow(['Trial number (down)'])
                        for mc_trial in range(cumulative_dvh_counts_by_MC_trial_arr.shape[0]):
                            if mc_trial == 0:
                                mc_trial_desc = 'Nominal'
                            else:
                                mc_trial_desc = str(mc_trial)
                            write.writerow([mc_trial_desc]+cumulative_dvh_counts_by_MC_trial_arr[mc_trial,:].tolist())

                    write.writerow(['___'])
                    write.writerow(['DVH metrics, percentages are relative to CTV target dose'])
                    write.writerow(['Each row is a fixed DVH metric, each column is a fixed MC trial'])
                    for vol_DVH_percent in v_percent_DVH_to_calc_list:
                        dvh_metric_all_MC_trials = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["All MC trials list"]
                        dvh_metric_nominal = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Nominal"]
                        dvh_metric_mean = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Mean"]
                        dvh_metric_std = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["STD"]
                        dvh_metric_quantiles_dict = dvh_metric_vol_dose_percent_dict[str(vol_DVH_percent)]["Quantiles"]
                        all_MC_trial_number_list = np.arange(1,len(dvh_metric_all_MC_trials)).tolist()
                        nominal_and_all_MC_trial_number_list = ["Nominal"]+all_MC_trial_number_list
                        write.writerow([' ']) 
                        write.writerow(['Trial number (across)']+nominal_and_all_MC_trial_number_list)
                        write.writerow(['DVH quantity (down)'])
                        write.writerow(['V'+str(vol_DVH_percent)+'%']+dvh_metric_all_MC_trials)
                        write.writerow(['V'+str(vol_DVH_percent)+'% mean', dvh_metric_mean]) 
                        write.writerow(['V'+str(vol_DVH_percent)+'% STD', dvh_metric_std])
                        for q,q_val in dvh_metric_quantiles_dict.items():
                            write.writerow(['V'+str(vol_DVH_percent)+'% '+str(q), q_val])

            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_completed_task, advance = 1)

        patients_progress.update(processing_patients_task, visible = False)
        completed_progress.update(processing_patients_completed_task, visible = True)


        patientUID_default = "Initializing"
        processing_patient_csv_writing_voxelized_description = "Writing dosimetry CSVs (voxelized) to file [{}]...".format(patientUID_default)
        processing_patients_task = patients_progress.add_task("[red]"+processing_patient_csv_writing_voxelized_description, total = master_structure_info_dict["Global"]["Num cases"])
        processing_patient_csv_writing_voxelized_description_completed = "Writing dosimetry CSVs (voxelized) to file"
        processing_patients_completed_task = completed_progress.add_task("[green]"+processing_patient_csv_writing_voxelized_description_completed, total=master_structure_info_dict["Global"]["Num cases"], visible=False)
        
        for patientUID,pydicom_item in master_structure_reference_dict.items():

            processing_patient_csv_writing_voxelized_description = "Writing dosimetry CSVs (voxelized) to file [{}]...".format(patientUID)
            patients_progress.update(processing_patients_task, description = "[red]" + processing_patient_csv_writing_voxelized_description)

            patient_sp_output_csv_dir = patient_sp_output_csv_dir_dict[patientUID]
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                dose_voxelized_output_file_name = patientUID+','+specific_bx_structure['ROI']+',n_MC_d='+str(num_MC_dose_simulations_input)+',n_bx='+str(num_sample_pts_per_bx)+'-dose_voxelized_out.csv'
                dose_voxelized_output_csv_file_path = patient_sp_output_csv_dir.joinpath(dose_voxelized_output_file_name)
                with open(dose_voxelized_output_csv_file_path, 'w', newline='') as f:
                    write = csv.writer(f)
                    write.writerow(['Patient ID ->',patientUID])
                    write.writerow(['BX ID ->',specific_bx_structure['ROI']])
                    write.writerow(['BX length (from contour data) (mm)', specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']])
                    write.writerow(['Num MC dose sims ->',num_MC_dose_simulations_input])
                    write.writerow(['Num bx pt samples ->',num_sample_pts_per_bx])
                    write.writerow(['Row ->','Info'])
                    write.writerow(['Col ->','Fixed voxel'])
                                                
                    voxelized_dose_dict = specific_bx_structure['MC data: voxelized dose results dict (dict of lists)']
                    num_voxels = voxelized_dose_dict["Num voxels"]
                    voxel_index_row = list(range(num_voxels))
                    voxel_index_row.insert(0,'Voxel index')
                    biopsy_z_voxel_range_row = voxelized_dose_dict["Voxel z range"].copy()
                    rounded_biopsy_z_voxel_range_row = [[round(sub_list[0],2),round(sub_list[1],2)] for sub_list in biopsy_z_voxel_range_row]
                    biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                    rounded_biopsy_z_voxel_range_row.insert(0,'Voxel z range (mm)')
                    num_sample_pts_in_voxel_row = voxelized_dose_dict["Num sample pts in voxel"].copy()
                    num_sample_pts_in_voxel_row.insert(0, 'Num sample pts in vxl')
                    num_MC_trials_in_voxel_row = voxelized_dose_dict['Total num MC trials in voxel'].copy()
                    num_MC_trials_in_voxel_row.insert(0, 'Num MC trials in vxl')
                    arth_mean_dose_row = voxelized_dose_dict["Arithmetic mean of dose in voxel"].copy()
                    arth_mean_dose_row.insert(0, 'Arth mean (dose)')
                    std_dev_dose_row = voxelized_dose_dict["Std dev of dose in voxel"].copy()
                    std_dev_dose_row.insert(0, 'Std dev (binomial estimator)')

                    write.writerow(voxel_index_row)
                    write.writerow(biopsy_z_voxel_range_row)
                    write.writerow(rounded_biopsy_z_voxel_range_row)
                    write.writerow(num_sample_pts_in_voxel_row)
                    write.writerow(num_MC_trials_in_voxel_row)
                    write.writerow(arth_mean_dose_row)
                    write.writerow(std_dev_dose_row)
                    for i in range(5):
                        write.writerow([''])
                    
                    write.writerow(['Row ->','Fixed voxel'])
                    write.writerow(['Col ->','Dose values'])
                    
                    voxelized_dose_list = specific_bx_structure['MC data: voxelized dose results list']

                    for voxel_index, voxel_dict in enumerate(voxelized_dose_list):
                        dose_vals_in_voxel_row = voxel_dict['All dose vals in voxel list'].copy()
                        dose_vals_in_voxel_row.insert(0,voxel_index)
                        write.writerow(dose_vals_in_voxel_row)

            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_completed_task, advance = 1)

        patients_progress.update(processing_patients_task, visible = False)
        completed_progress.update(processing_patients_completed_task, visible = True)