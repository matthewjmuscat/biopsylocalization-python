import numpy as np
import pandas 

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