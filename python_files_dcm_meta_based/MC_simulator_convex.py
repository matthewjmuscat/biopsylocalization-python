import time
import biopsy_creator
import loading_tools # imported for more sophisticated loading bar
import numpy as np
import open3d as o3d
import point_containment_tools
import plotting_funcs
import rich
from rich.progress import Progress, track
import time 
import sys
import math_funcs as mf
import scipy
import math
import statistics
import copy


def simulator(master_structure_reference_dict, structs_referenced_list, num_simulations):

    ref_list = ["Bx ref","OAR ref","DIL ref"] # note that Bx ref has to be the first entry for other parts of the code to work!
    uncertainty_sources_list = []
    uncertainty_sources_raw_dict = {}

    num_biopsies = sum([len(patient_dict[1][ref_list[0]]) for patient_dict in master_structure_reference_dict.items()])

    print('Number of biopsy tracks to simulate are: ', num_biopsies,'.')
    print('Number of simulations per biopsy track has been set to: ', num_simulations,'.')
    

    st = time.time()
    with loading_tools.Loader(num_biopsies,"Reconstructing and sampling biopsies...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            Bx_structs = structs_referenced_list[0]
            for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]):
                centroid_line = specific_BX_structure["Best fit line of centroid pts"]
                origin_to_first_centroid_vector = specific_BX_structure["Centroid line sample pts"][0]
                list_origin_to_first_centroid_vector = np.squeeze(origin_to_first_centroid_vector).tolist()
                biopsy_samples = biopsy_creator.biopsy_points_reconstruction_and_uniform_sampler(list_origin_to_first_centroid_vector,centroid_line)
                biopsy_samples_point_cloud = o3d.geometry.PointCloud()
                biopsy_samples_point_cloud.points = o3d.utility.Vector3dVector(biopsy_samples[:,0:3])
                pcd_color = np.random.uniform(0, 0.7, size=3)
                biopsy_samples_point_cloud.paint_uniform_color(pcd_color)
                master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Random uniformly sampled volume pts arr"] = biopsy_samples
                biopsy_raw_point_cloud = master_structure_reference_dict[patientUID][Bx_structs][specific_BX_structure_index]["Point cloud raw"]
                pcd_color = np.random.uniform(0, 0.7, size=3)
                biopsy_raw_point_cloud.paint_uniform_color(pcd_color)

                # plot point clouds?
                o3d.visualization.draw_geometries([biopsy_raw_point_cloud,biopsy_samples_point_cloud])



    with loading_tools.Loader(num_biopsies*num_simulations,"Simulating biopsy uncertainties...") as loader:
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            Bx_structs = structs_referenced_list[0]
            for specific_BX_structure_index, specific_BX_structure in enumerate(pydicom_item[Bx_structs]):
                structure_uncertainty_array = np.empty([num_simulations, specific_BX_structure["Uncertainty params"].size])
                for mu,sigma in specific_BX_structure["Uncertainty params"]: 
                    specific_uncertainty_array = np.random.normal(mu, sigma, num_simulations)

                    for j in range(0, num_simulations):
                        print(1)
                        
    return master_structure_reference_dict



def simulator_parallel(parallel_pool, 
                       live_display, 
                       layout_groups, 
                       master_structure_reference_dict, 
                       structs_referenced_list, 
                       dose_ref, 
                       master_structure_info_dict, 
                       biopsy_z_voxel_length, 
                       num_dose_calc_NN, 
                       spinner_type):
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    with live_display:
        live_display.start(refresh = True)
        num_patients = master_structure_info_dict["Global"]["Num patients"]
        num_global_structures = master_structure_info_dict["Global"]["Num structures"]
        num_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC simulations"]
        num_sample_pts_per_bx = master_structure_info_dict["Global"]["MC info"]["Num sample pts per BX core"]



        default_output = "Initializing"
        processing_patients_task_main_description = "[red]Generating {} MC samples for {} structures [{}]...".format(num_simulations,num_global_structures,default_output)
        processing_patients_task_completed_main_description = "[green]Generating {} MC samples for {} structures".format(num_simulations,num_global_structures)
        processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total = num_patients)
        processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total = num_patients, visible = False)

        # simulate all structure shifts in parallel and update the master reference dict
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            processing_patients_task_main_description = "[red]Generating {} MC samples for {} structures [{}]...".format(num_simulations,num_global_structures,patientUID)
            patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)


            patient_dict_updated_with_all_structs_generated_norm_dist_translation_samples = MC_simulator_shift_all_structures_generator_parallel(parallel_pool, pydicom_item, structs_referenced_list, num_simulations)
            master_structure_reference_dict[patientUID] = patient_dict_updated_with_all_structs_generated_norm_dist_translation_samples
            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_task_completed, advance = 1)
        patients_progress.update(processing_patients_task, visible = False, refresh = True)
        completed_progress.update(processing_patients_task_completed, visible = True, refresh = True)
        live_display.refresh()

        
        num_biopsies_global = master_structure_info_dict["Global"]["Num biopsies"]
        num_OARs_global = master_structure_info_dict["Global"]["Num OARs"]
        num_DILs_global = master_structure_info_dict["Global"]["Num DILs"]
        
        simulation_info_important_line_str = "Simulation data: # MC samples = {} | # sample pts per BX core = {} | # biopsies = {} | # anatomical structures = {} | # patients = {}.".format(str(num_simulations), str(num_sample_pts_per_bx), str(num_biopsies_global), str(num_global_structures-num_biopsies_global), str(num_patients))
        important_info.add_text_line(simulation_info_important_line_str, live_display)
        

        default_patientUID = "initializing"
        translating_patients_main_desc = "[red]MC simulating biopsy and anatomy randomized translations [{}]...".format(default_patientUID)
        translating_patients_structures_task = patients_progress.add_task(translating_patients_main_desc, total=num_patients)
        translating_patients_structures_task_completed = completed_progress.add_task("[green]MC simulating biopsy and anatomy randomized translations", total=num_patients, visible = False)
        # simulate every biopsy sequentially
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            translating_patients_main_desc = "[red]MC simulating biopsy and anatomy randomized translations [{}]...".format(patientUID)
            patients_progress.update(translating_patients_structures_task, description = translating_patients_main_desc)

            # create a dictionary of all non bx structures
            structure_organized_for_bx_data_blank_dict = {}
            for non_bx_struct_type in structs_referenced_list[1:]:
                for specific_non_bx_structure_index, specific_non_bx_structure in enumerate(pydicom_item[non_bx_struct_type]):
                    specific_non_bx_struct_roi = specific_non_bx_structure["ROI"]
                    specific_non_bx_struct_refnum = specific_non_bx_structure["Ref #"]
                    structure_organized_for_bx_data_blank_dict[specific_non_bx_struct_roi,non_bx_struct_type,specific_non_bx_struct_refnum,specific_non_bx_structure_index] = None
            #MC_translation_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
            #MC_compiled_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
            # set structure type to BX 
            bx_structure_type = structs_referenced_list[0]
            local_patient_num_biopsies = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            translating_bx_and_structure_relative_main_desc = "[cyan]~For each biopsy [{},{}]...".format(patientUID, "initializing")
            translating_biopsy_relative_to_structures_task = biopsies_progress.add_task(translating_bx_and_structure_relative_main_desc, total=local_patient_num_biopsies)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                translating_bx_and_structure_relative_main_desc = "[cyan]~For each biopsy [{},{}]...".format(patientUID, specific_bx_structure_roi)
                biopsies_progress.update(translating_biopsy_relative_to_structures_task, description = translating_bx_and_structure_relative_main_desc)
                
                # Do all trials in parallel
                indeterminate_sub_desc_bx_shift = "[cyan]~~Shifting biopsy structure (BX shift) [{},{}]".format(patientUID, specific_bx_structure_roi)
                indeterminate_sub_bx_shift_task = indeterminate_progress_sub.add_task(indeterminate_sub_desc_bx_shift, total=None)
                bx_only_shifted_randomly_sampled_bx_pts_3Darr = MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_parallel(parallel_pool, specific_bx_structure)
                # THIS SHOULD BE SAVED AT THE END. Save the 3d array of the bx only shifted data containing all MC trials as slices to the master reference dictionary
                indeterminate_progress_sub.update(indeterminate_sub_bx_shift_task, visible = False)
                master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: bx only shifted 3darr"] = bx_only_shifted_randomly_sampled_bx_pts_3Darr
                
                indeterminate_sub_desc_bx_shift = "[cyan]~~Shifting biopsy structure (relative OAR and DIL shifts) [{},{}]".format(patientUID, specific_bx_structure_roi)
                indeterminate_sub_bx_shift_task = indeterminate_progress_sub.add_task(indeterminate_sub_desc_bx_shift, total=None)
                structure_shifted_bx_data_dict = MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_parallel(parallel_pool, pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, structure_organized_for_bx_data_blank_dict)
                master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: bx and structure shifted dict"] = structure_shifted_bx_data_dict
                indeterminate_progress_sub.update(indeterminate_sub_bx_shift_task, visible = False)

                biopsies_progress.update(translating_biopsy_relative_to_structures_task, advance = 1)
            biopsies_progress.update(translating_biopsy_relative_to_structures_task, visible = False)
            
            patients_progress.update(translating_patients_structures_task, advance=1)
            completed_progress.update(translating_patients_structures_task_completed, advance=1)
        patients_progress.update(translating_patients_structures_task, visible=False)
        completed_progress.update(translating_patients_structures_task_completed, visible=True)
        live_display.refresh()


        
        testing_biopsy_containment_patient_task = patients_progress.add_task("[red]Testing biopsy containment in all anatomical structures...", total=num_patients)
        testing_biopsy_containment_patient_task_completed = completed_progress.add_task("[green]Testing biopsy containment in all anatomical structures", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            patients_progress.update(testing_biopsy_containment_patient_task, description = "[red]Testing biopsy containment in all anatomical structures [{}]...".format(patientUID))
            bx_structure_type = structs_referenced_list[0]
            testing_biopsy_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID,"initializing"), total = sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                bx_specific_biopsy_containment_desc = "[cyan]~For each biopsy [{},{}]...".format(patientUID, specific_bx_structure_roi)
                biopsies_progress.update(testing_biopsy_containment_task, description = bx_specific_biopsy_containment_desc)
                
                # paint the unshifted bx sampled points purple for later viewing
                unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                testing_each_non_bx_structure_containment_task = structures_progress.add_task("[cyan]~~For each non-BX structure [{},{},{}]...".format(patientUID,specific_bx_structure_roi,"initializing"), total=sp_patient_total_num_non_BXs)
                structure_shifted_bx_data_dict = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: bx and structure shifted dict"] 
                MC_translation_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                for structure_info,shifted_bx_data_3darr in structure_shifted_bx_data_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_refnum = structure_info[2]
                    structure_index = structure_info[3]

                    structures_progress.update(testing_each_non_bx_structure_containment_task, description = "[cyan]~~For each non-BX structure [{},{},{}]...".format(patientUID,specific_bx_structure_roi,structure_roi))

                    non_bx_struct_deulaunay_objs_zslice_wise_list = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Delaunay triangulation zslice-wise list"] 
                    non_bx_struct_deulaunay_obj_global_convex = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Delaunay triangulation global structure"] 
                    non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                    non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                    non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))
                    prostate_interslice_interpolation_information = master_structure_reference_dict[patientUID]['OAR ref'][0]["Inter-slice interpolation information"]
                    prostate_interpolated_pts_np_arr = prostate_interslice_interpolation_information.interpolated_pts_np_arr
                    prostate_interpolated_pts_pcd = point_containment_tools.create_point_cloud(prostate_interpolated_pts_np_arr, color = np.array([0,1,1]))
                    
                    testing_each_trial_task = MC_trial_progress.add_task("[cyan]~~~For each MC trial [{},{},{}]...".format(patientUID,specific_bx_structure_roi,structure_roi), total=num_simulations)
                    all_trials_POP_test_results_and_point_clouds_tuple = []
                    for single_trial_shifted_bx_data_arr in shifted_bx_data_3darr:
                        #single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple = point_containment_test_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, non_bx_struct_deulaunay_obj_global_convex, non_bx_struct_interslice_interpolation_information, single_trial_shifted_bx_data_arr)
                        single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple = point_containment_test_axis_aligned_bounding_box_and_zslice_wise_2d_PIP_parallel(parallel_pool, num_simulations, non_bx_struct_interpolated_pts_np_arr, non_bx_struct_interslice_interpolation_information, single_trial_shifted_bx_data_arr)
                        all_trials_POP_test_results_and_point_clouds_tuple.append(single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple)
                        
                        
                        # plot results to make sure everything is working properly, containment structure is blue, shifted and tested pcd should be red and green
                        """
                        if non_bx_structure_type == 'DIL ref':
                            bx_test_pts_results = single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple[1]
                            bx_test_pts_results_pcd = single_trial_shifted_bx_data_results_fully_concave_and_point_cloud_tuple[1]
                            structure_and_bx_shifted_bx_pcd = point_containment_tools.create_point_cloud(single_trial_shifted_bx_data_arr)
                            plotting_funcs.plot_geometries(bx_test_pts_results_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, label='Unknown')
                            plotting_funcs.plot_geometries(bx_test_pts_results_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd, label='Unknown')
                            plotting_funcs.plot_two_views_side_by_side([bx_test_pts_results_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd], "ScreenCamera_2023-02-19-15-14-47.json", [bx_test_pts_results_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd], "ScreenCamera_2023-02-19-15-27-46.json")
                            plotting_funcs.plot_two_views_side_by_side([bx_test_pts_results_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd], "ScreenCamera_2023-02-19-15-14-47.json", [bx_test_pts_results_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd], "ScreenCamera_2023-02-19-15-29-43.json")
                        """
                        
                        MC_trial_progress.update(testing_each_trial_task, advance=1)
                    
                    MC_trial_progress.remove_task(testing_each_trial_task)
                    MC_translation_results_for_fixed_bx_dict[structure_info] = all_trials_POP_test_results_and_point_clouds_tuple

                    structures_progress.update(testing_each_non_bx_structure_containment_task, advance=1)
                
                structures_progress.remove_task(testing_each_non_bx_structure_containment_task)
                # Update the master dictionary
                master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: MC sim translation results dict"] = MC_translation_results_for_fixed_bx_dict


                biopsies_progress.update(testing_biopsy_containment_task, advance=1)
            biopsies_progress.remove_task(testing_biopsy_containment_task)

            patients_progress.update(testing_biopsy_containment_patient_task, advance = 1)
            completed_progress.update(testing_biopsy_containment_patient_task_completed, advance = 1)
        patients_progress.update(testing_biopsy_containment_patient_task, visible = False)
        completed_progress.update(testing_biopsy_containment_patient_task_completed, visible = True)
        live_display.refresh()

        structure_specific_results_dict_empty = {"Total successes (containment) list": None, "Binomial estimator list": None, "Confidence interval 95 (containment) list": None, "Standard error (containment) list": None}
        compiling_results_patient_containment_task = patients_progress.add_task("[red]Compiling MC results ...", total=num_patients)
        compiling_results_patient_containment_task_completed = completed_progress.add_task("[green]Compiling MC results", total=num_patients, visible = False)  
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(compiling_results_patient_containment_task, description = "[red]Compiling MC results [{}]...".format(patientUID), total=num_patients)
            bx_structure_type = structs_referenced_list[0]           
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            compiling_results_biopsy_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID,"initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(compiling_results_biopsy_containment_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID,specific_bx_structure_roi))
                MC_translation_results_for_fixed_bx_dict = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: MC sim translation results dict"] 
                MC_compiled_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
                sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs
                compiling_results_each_non_bx_structure_containment_task = structures_progress.add_task("[cyan]~~For each structure [{},{},{}]...".format(patientUID,specific_bx_structure_roi,"initializing"), total=sp_patient_total_num_non_BXs)
                for structureID,structure_MC_results in MC_translation_results_for_fixed_bx_dict.items():
                    structures_progress.update(compiling_results_each_non_bx_structure_containment_task, description = "[cyan]~~For each structure [{},{},{}]...".format(patientUID,specific_bx_structure_roi,structureID), total=sp_patient_total_num_non_BXs)
                    structure_specific_results_dict = structure_specific_results_dict_empty.copy()
                    # counter list needs to be reset for every structure 
                    bx_containment_counter_by_org_pt_ind_list = [0]*num_sample_pts_per_bx    
                    compiling_results_each_trial_task = MC_trial_progress.add_task("[cyan]~~~For each MC trial...", total=num_simulations)
                    for MC_trial in structure_MC_results:
                        MC_trial_BX_pts_result_list = MC_trial[0]
                        for bx_pt_index, bx_point_result in enumerate(MC_trial_BX_pts_result_list):
                            pt_contained = None
                            if bx_point_result[0] == None:
                                pt_contained = False
                            elif bx_point_result[0][0] == False:
                                pt_contained = False
                            elif bx_point_result[0][0] == True:
                                pt_contained = True
                            else:
                                print('Something went wrong!')
                                sys.exit('Programme exited.')
                            if pt_contained == True:
                                bx_containment_counter_by_org_pt_ind_list[bx_pt_index] = bx_containment_counter_by_org_pt_ind_list[bx_pt_index] + 1
                            else: 
                                pass 
                        MC_trial_progress.update(compiling_results_each_trial_task, advance=1) 
                    MC_trial_progress.remove_task(compiling_results_each_trial_task)
                    structure_specific_results_dict["Total successes (containment) list"] = bx_containment_counter_by_org_pt_ind_list
                    bx_containment_binomial_estimator_by_org_pt_ind_list = [x/num_simulations for x in bx_containment_counter_by_org_pt_ind_list]
                    structure_specific_results_dict["Binomial estimator list"] = bx_containment_binomial_estimator_by_org_pt_ind_list
                    MC_compiled_results_for_fixed_bx_dict[structureID] = structure_specific_results_dict
                    structures_progress.update(compiling_results_each_non_bx_structure_containment_task, advance=1)
                structures_progress.remove_task(compiling_results_each_non_bx_structure_containment_task)
                biopsies_progress.update(compiling_results_biopsy_containment_task, advance = 1) 
                master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: compiled sim results"] = MC_compiled_results_for_fixed_bx_dict
            biopsies_progress.remove_task(compiling_results_biopsy_containment_task) 
            patients_progress.update(compiling_results_patient_containment_task, advance = 1) 
            completed_progress.update(compiling_results_patient_containment_task_completed, advance = 1)
        patients_progress.update(compiling_results_patient_containment_task, visible = False) 
        completed_progress.update(compiling_results_patient_containment_task_completed, visible = True)
        live_display.refresh()

        
        calc_MC_stat_biopsy_containment_task = patients_progress.add_task("[red]Calculating MC statistics [{}]...".format("initializing"), total=num_patients)
        calc_MC_stat_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating MC statistics", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_MC_stat_biopsy_containment_task, description = "[red]Calculating MC statistics [{}]...".format(patientUID))
            bx_structure_type = structs_referenced_list[0]           
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            calc_MC_stat_each_bx_structure_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID,"initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_results_dict = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: compiled sim results"] 
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(calc_MC_stat_each_bx_structure_containment_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID,specific_bx_structure_roi))
                
                for structureID,structure_specific_results_dict in specific_bx_results_dict.items():
                    bx_containment_binomial_estimator_by_org_pt_ind_list = structure_specific_results_dict["Binomial estimator list"]
                    bx_containment_counter_by_org_pt_ind_list = structure_specific_results_dict["Total successes (containment) list"] 
                    probability_estimator_list = bx_containment_binomial_estimator_by_org_pt_ind_list
                    num_successes_list = bx_containment_counter_by_org_pt_ind_list
                    num_trials = num_simulations
                    confidence_interval_list = calculate_binomial_containment_conf_intervals_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials)
                    standard_err_list = calculate_binomial_containment_stand_err_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials)
                    structure_specific_results_dict["Confidence interval 95 (containment) list"] = confidence_interval_list
                    structure_specific_results_dict["Standard error (containment) list"] = standard_err_list

                    
                biopsies_progress.update(calc_MC_stat_each_bx_structure_containment_task, advance = 1)
            biopsies_progress.remove_task(calc_MC_stat_each_bx_structure_containment_task)
            patients_progress.update(calc_MC_stat_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_MC_stat_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_MC_stat_biopsy_containment_task, visible = False)
        completed_progress.update(calc_MC_stat_biopsy_containment_task_complete,visible = True)
        live_display.refresh()

        
        # voxelize containment results
        biopsy_voxelize_containment_task = patients_progress.add_task("[red]Voxelizing containment results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_containment_task_complete = completed_progress.add_task("[green]Voxelizing containment results", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing containment results [{}]...".format(patientUID))
            bx_structure_type = structs_referenced_list[0]           
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID,"initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_results_dict = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: compiled sim results"] 
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_containment_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID,specific_bx_structure_roi))
                
                voxelized_containment_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                voxelized_containment_results_for_fixed_bx_dict_alt = structure_organized_for_bx_data_blank_dict.copy()
                randomly_sampled_bx_pts_bx_coord_sys_arr = specific_bx_structure['Random uniformly sampled volume pts bx coord sys arr']
                biopsy_cyl_z_length = specific_bx_structure["Reconstructed biopsy cylinder length (from contour data)"]
                #rounded_down_biopsy_cyl_z_length = float(math.floor(biopsy_cyl_z_length))
                num_z_voxels = float(math.floor(float(biopsy_cyl_z_length/biopsy_z_voxel_length)))
                constant_voxel_biopsy_cyl_z_length = num_z_voxels*biopsy_z_voxel_length
                biopsy_z_length_difference = biopsy_cyl_z_length - constant_voxel_biopsy_cyl_z_length
                extra_length_for_biopsy_end_cap_voxels = biopsy_z_length_difference/2
                

                for structureID,structure_specific_results_dict in specific_bx_results_dict.items():
                    binomial_estimator_list = structure_specific_results_dict["Binomial estimator list"]
                    total_success_list = structure_specific_results_dict["Total successes (containment) list"]
                    conf_interval_list = structure_specific_results_dict["Confidence interval 95 (containment) list"]
                    
                    voxel_z_begin = 0.
                    voxelized_biopsy_containment_results_list = [None]*int(num_z_voxels)
                    voxel_dict_empty = {"Voxel z begin": None, "Voxel z end": None, "Indices from all sample pts that are within voxel arr": None, "Num sample pts in voxel": None, "Sample pts in voxel arr (bx coord sys)": None, "Total successes in voxel list": None, "Total successes in voxel": None, "Total num MC trials in voxel": None, "Binomial estimators in voxel list": None, "Arithmetic mean of binomial estimators in voxel": None, "Std dev of binomial estimators in voxel": None, "Conf interval in voxel list": None}
                    for voxel_index in range(int(num_z_voxels)):
                        if voxel_index == 0 or voxel_index == range(int(num_z_voxels))[-1]:
                            voxel_z_end = voxel_z_begin + biopsy_z_voxel_length + extra_length_for_biopsy_end_cap_voxels
                        else:
                            voxel_z_end = voxel_z_begin + biopsy_z_voxel_length
                            
                        # find indices of the points in the biopsy that fall between the voxel bounds
                        sample_pts_indices_in_voxel_arr = np.asarray(np.logical_and(randomly_sampled_bx_pts_bx_coord_sys_arr[:,2] >= voxel_z_begin , randomly_sampled_bx_pts_bx_coord_sys_arr[:,2] <= voxel_z_end)).nonzero()
                        num_sample_pts_in_voxel = sample_pts_indices_in_voxel_arr[0].shape[0]
                        samples_pts_in_voxel_arr = np.take(randomly_sampled_bx_pts_bx_coord_sys_arr, sample_pts_indices_in_voxel_arr, axis=0)[0]
                        binomial_estimator_in_voxel_list = np.take(binomial_estimator_list, sample_pts_indices_in_voxel_arr)[0].tolist()
                        total_success_in_voxel_list = np.take(total_success_list, sample_pts_indices_in_voxel_arr)[0].tolist()
                        conf_interval_in_voxel_list = np.take(conf_interval_list, sample_pts_indices_in_voxel_arr, axis = 0)[0].tolist()

                        total_successes_in_voxel = sum(total_success_in_voxel_list)
                        total_num_MC_trials_in_voxel = num_sample_pts_in_voxel*num_simulations
                        if num_sample_pts_in_voxel < 1:
                            arithmetic_mean_binomial_estimators_in_voxel = 'No data'
                        else:
                            arithmetic_mean_binomial_estimators_in_voxel = statistics.mean(binomial_estimator_in_voxel_list)
                        if num_sample_pts_in_voxel <= 1:
                            std_dev_binomial_estimators_in_voxel = 0
                        else:
                            std_dev_binomial_estimators_in_voxel = statistics.stdev(binomial_estimator_in_voxel_list)

                        voxel_dict = voxel_dict_empty.copy()
                        voxel_dict["Voxel z begin"] = voxel_z_begin
                        voxel_dict["Voxel z end"] = voxel_z_end
                        voxel_dict["Indices from all sample pts that are within voxel arr"] = sample_pts_indices_in_voxel_arr
                        voxel_dict["Num sample pts in voxel"] = num_sample_pts_in_voxel
                        voxel_dict["Sample pts in voxel arr (bx coord sys)"] = samples_pts_in_voxel_arr
                        voxel_dict["Total successes in voxel list"] = total_success_in_voxel_list
                        voxel_dict["Total successes in voxel"] = total_successes_in_voxel
                        voxel_dict["Total num MC trials in voxel"] = total_num_MC_trials_in_voxel
                        voxel_dict["Binomial estimators in voxel list"] = binomial_estimator_in_voxel_list
                        voxel_dict["Arithmetic mean of binomial estimators in voxel"] = arithmetic_mean_binomial_estimators_in_voxel
                        voxel_dict["Std dev of binomial estimators in voxel"] = std_dev_binomial_estimators_in_voxel
                        voxel_dict["Conf interval in voxel list"] = conf_interval_in_voxel_list

                        voxelized_biopsy_containment_results_list[voxel_index] = voxel_dict

                        voxel_z_begin = voxel_z_end
                    
                    voxelized_containment_results_for_fixed_bx_dict[structureID] = voxelized_biopsy_containment_results_list
                    
                    # reorganize this data in a better way (didnt want to delete/change above code), but better to have a dictionary of lists rather than a list of dictionaries
                    voxel_dict_of_lists = voxel_dict_empty.copy()
                    #voxel_dict_of_lists = dict.fromkeys(voxel_dict_of_lists,[])
                    for key,value in voxel_dict_of_lists.items():
                        voxel_dict_of_lists[key] = []
                    voxel_dict_of_lists["Voxel z range"] = []
                    voxel_dict_of_lists["Voxel z range rounded"] = []
                    for voxel_index in range(int(num_z_voxels)):
                        voxel_dict = voxelized_biopsy_containment_results_list[voxel_index]
                        voxel_dict_of_lists["Voxel z begin"].append(voxel_dict["Voxel z begin"])
                        voxel_dict_of_lists["Voxel z end"].append(voxel_dict["Voxel z end"])
                        voxel_dict_of_lists["Voxel z range"].append([voxel_dict["Voxel z begin"],voxel_dict["Voxel z end"]])
                        voxel_dict_of_lists["Voxel z range rounded"].append([round(voxel_dict["Voxel z begin"],2),round(voxel_dict["Voxel z end"],2)])
                        voxel_dict_of_lists["Indices from all sample pts that are within voxel arr"].append(voxel_dict["Indices from all sample pts that are within voxel arr"])
                        voxel_dict_of_lists["Num sample pts in voxel"].append(voxel_dict["Num sample pts in voxel"])
                        voxel_dict_of_lists["Sample pts in voxel arr (bx coord sys)"].append(voxel_dict["Sample pts in voxel arr (bx coord sys)"])
                        voxel_dict_of_lists["Total successes in voxel list"].append(voxel_dict["Total successes in voxel list"])
                        voxel_dict_of_lists["Total successes in voxel"].append(voxel_dict["Total successes in voxel"])
                        voxel_dict_of_lists["Total num MC trials in voxel"].append(voxel_dict["Total num MC trials in voxel"])
                        voxel_dict_of_lists["Binomial estimators in voxel list"].append(voxel_dict["Binomial estimators in voxel list"])
                        voxel_dict_of_lists["Arithmetic mean of binomial estimators in voxel"].append(voxel_dict["Arithmetic mean of binomial estimators in voxel"])
                        voxel_dict_of_lists["Std dev of binomial estimators in voxel"].append(voxel_dict["Std dev of binomial estimators in voxel"])
                        voxel_dict_of_lists["Conf interval in voxel list"].append(voxel_dict["Conf interval in voxel list"])
                        
                    voxel_dict_of_lists["Num voxels"] = int(num_z_voxels)
                    voxelized_containment_results_for_fixed_bx_dict_alt[structureID] = voxel_dict_of_lists

                specific_bx_structure["MC data: voxelized containment results dict"] = voxelized_containment_results_for_fixed_bx_dict
                specific_bx_structure["MC data: voxelized containment results dict (dict of lists)"] = voxelized_containment_results_for_fixed_bx_dict_alt
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_containment_task, advance = 1)
            biopsies_progress.remove_task(biopsy_voxelize_each_bx_structure_containment_task)
            patients_progress.update(biopsy_voxelize_containment_task, advance = 1)
            completed_progress.update(biopsy_voxelize_containment_task_complete, advance = 1)
        patients_progress.update(biopsy_voxelize_containment_task, visible = False)
        completed_progress.update(biopsy_voxelize_containment_task_complete,visible = True)
        live_display.refresh()

        

        
        calc_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]Calculating NN dosimetric localization [{}]...".format("initializing"), total=num_patients)
        calc_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating NN dosimetric localization", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_dose_NN_biopsy_containment_task, description = "[red]Calculating NN dosimetric localization [{}]...".format(patientUID))
            # create KDtree for dose data
            dose_ref_dict = pydicom_item[dose_ref]
            phys_space_dose_map_3d_arr = dose_ref_dict["Dose phys space and pixel 3d arr"]
            phys_space_dose_map_3d_arr_flattened = np.reshape(phys_space_dose_map_3d_arr, (-1,7) , order = 'C') # turn the data into a 2d array
            phys_space_dose_map_phys_coords_2d_arr = phys_space_dose_map_3d_arr_flattened[:,3:6] 
            phys_space_dose_map_dose_2d_arr = phys_space_dose_map_3d_arr_flattened[:,6] 
            dose_data_KDtree = scipy.spatial.KDTree(phys_space_dose_map_phys_coords_2d_arr)
            dose_ref_dict["KDtree"] = dose_data_KDtree
            
            bx_structure_type = structs_referenced_list[0]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            dosimetric_calc_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID, "initializing"), total=sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(dosimetric_calc_biopsy_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID, specific_bx_structure_roi))
                
                bx_only_shifted_3darr = specific_bx_structure["MC data: bx only shifted 3darr"] # note that the 3rd dimension slices are each MC trial
                dosimetric_calc_parallel_task = indeterminate_progress_sub.add_task("[cyan]~~Conducting NN search [{},{}]...".format(patientUID, specific_bx_structure_roi), total = None)
                dosimetric_localization_all_MC_trials_list = dosimetric_localization_parallel(parallel_pool, bx_only_shifted_3darr, specific_bx_structure, dose_ref_dict, dose_ref, phys_space_dose_map_phys_coords_2d_arr, phys_space_dose_map_dose_2d_arr, num_dose_calc_NN)
                specific_bx_structure['MC data: bx to dose NN search objects list'] = dosimetric_localization_all_MC_trials_list
                indeterminate_progress_sub.remove_task(dosimetric_calc_parallel_task)

                biopsies_progress.update(dosimetric_calc_biopsy_task, advance=1)
            biopsies_progress.remove_task(dosimetric_calc_biopsy_task)
            patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_dose_NN_biopsy_containment_task, visible = False)
        completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, visible = True)
        live_display.refresh()
                    

        bx_structure_type = structs_referenced_list[0]
        compile_results_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]Compiling dosimetric localization results [{}]...".format("initializing"), total=num_patients)
        compile_results_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]Compiling dosimetric localization results", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(compile_results_dose_NN_biopsy_containment_task, description = "[red]Compiling dosimetric localization results [{}]...".format(patientUID))
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            compile_results_dose_NN_biopsy_containment_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID, "initializing"), total = sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID, specific_bx_structure_roi))
                dosimetric_localization_all_MC_trials_list = specific_bx_structure['MC data: bx to dose NN search objects list']
                dosimetric_localization_all_MC_trials_list_NN_lists_only = [NN_parent_obj.NN_data_list for NN_parent_obj in dosimetric_localization_all_MC_trials_list]
                dosimetric_localization_NN_child_objs_by_bx_point_all_trials_list = list(zip(*dosimetric_localization_all_MC_trials_list_NN_lists_only))
                dosimetric_localization_dose_vals_by_bx_point_all_trials_list = [[NN_child_obj.nearest_dose for NN_child_obj in fixed_bx_pt_NN_objs_list] for fixed_bx_pt_NN_objs_list in dosimetric_localization_NN_child_objs_by_bx_point_all_trials_list]

                specific_bx_structure["MC data: Dose NN child obj for each sampled bx pt list"] = dosimetric_localization_NN_child_objs_by_bx_point_all_trials_list
                specific_bx_structure["MC data: Dose vals for each sampled bx pt list"] = dosimetric_localization_dose_vals_by_bx_point_all_trials_list

                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, advance = 1)
            biopsies_progress.remove_task(compile_results_dose_NN_biopsy_containment_by_biopsy_task)    
            patients_progress.update(compile_results_dose_NN_biopsy_containment_task, advance=1)
            completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, advance=1)
        patients_progress.update(compile_results_dose_NN_biopsy_containment_task, visible = False)
        completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, visible = True)
        live_display.refresh()




        bx_structure_type = structs_referenced_list[0]
        computing_MLE_statistics_dose_task = patients_progress.add_task("[red]Computing dosimetric localization statistics (MLE) [{}]...".format("initializing"), total=num_patients)
        computing_MLE_statistics_dose_task_complete = completed_progress.add_task("[green]Computing dosimetric localization statistics (MLE)", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(computing_MLE_statistics_dose_task, description = "[red]Computing dosimetric localization statistics (MLE) [{}]...".format(patientUID))
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            compile_results_dose_NN_biopsy_containment_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID, "initializing"), total = sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID, specific_bx_structure_roi))
                dosimetric_localization_dose_vals_by_bx_point_all_trials_list = specific_bx_structure["MC data: Dose vals for each sampled bx pt list"] 
                
                dosimetric_MLE_statistics_all_bx_pts_list = normal_distribution_MLE_parallel(parallel_pool, dosimetric_localization_dose_vals_by_bx_point_all_trials_list)
                mu_se_var_all_bx_pts_list = [bx_point_stats[0] for bx_point_stats in dosimetric_MLE_statistics_all_bx_pts_list]
                confidence_intervals_all_bx_pts_list = [bx_point_stats[1] for bx_point_stats in dosimetric_MLE_statistics_all_bx_pts_list]
                
                mu_all_bx_pts_list = np.mean(dosimetric_localization_dose_vals_by_bx_point_all_trials_list, axis = 1).tolist()
                std_all_bx_pts_list = np.std(dosimetric_localization_dose_vals_by_bx_point_all_trials_list, axis = 1, ddof=1).tolist()
                quantiles_all_bx_pts_dict_of_lists = {'Q'+str(q): np.quantile(dosimetric_localization_dose_vals_by_bx_point_all_trials_list, q/100,axis = 1).tolist() for q in range(5,100,5)}

                MC_dose_stats_dict = {"Dose statistics by bx pt (mean,se,var)": mu_se_var_all_bx_pts_list, "Confidence intervals (95%) by bx pt": confidence_intervals_all_bx_pts_list}
                MC_dose_stats_basic_dict = {"Mean dose by bx pt": mu_all_bx_pts_list, "STD by bx pt": std_all_bx_pts_list, "Qunatiles dose by bx pt dict": quantiles_all_bx_pts_dict_of_lists}
                specific_bx_structure["MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)"] = MC_dose_stats_dict
                specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std)"] = MC_dose_stats_basic_dict
                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, advance = 1)
            
            biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, visible = False)
            patients_progress.update(computing_MLE_statistics_dose_task, advance = 1)
            completed_progress.update(computing_MLE_statistics_dose_task_complete, advance = 1)

        patients_progress.update(computing_MLE_statistics_dose_task, visible = False)
        completed_progress.update(computing_MLE_statistics_dose_task_complete, visible = True)



        # voxelize dose results
        biopsy_voxelize_dose_task = patients_progress.add_task("[red]Voxelizing dose results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_dose_task_complete = completed_progress.add_task("[green]Voxelizing dose results", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing dose results [{}]...".format(patientUID))
            bx_structure_type = structs_referenced_list[0]           
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_dose_task = biopsies_progress.add_task("[cyan]~For each biopsy [{},{}]...".format(patientUID,"initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                specific_bx_dose_results_list = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]['MC data: Dose vals for each sampled bx pt list'] 
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, description = "[cyan]~For each biopsy [{},{}]...".format(patientUID,specific_bx_structure_roi))
                
                randomly_sampled_bx_pts_bx_coord_sys_arr = specific_bx_structure['Random uniformly sampled volume pts bx coord sys arr']
                biopsy_cyl_z_length = specific_bx_structure["Reconstructed biopsy cylinder length (from contour data)"]
                num_z_voxels = float(math.floor(float(biopsy_cyl_z_length/biopsy_z_voxel_length)))
                constant_voxel_biopsy_cyl_z_length = num_z_voxels*biopsy_z_voxel_length
                biopsy_z_length_difference = biopsy_cyl_z_length - constant_voxel_biopsy_cyl_z_length
                extra_length_for_biopsy_end_cap_voxels = biopsy_z_length_difference/2
                
                voxel_z_begin = 0.
                voxelized_biopsy_dose_results_list = [None]*int(num_z_voxels)
                voxel_dose_dict_empty = {"Voxel z begin": None, "Voxel z end": None, "Indices from all sample pts that are within voxel arr": None, "Num sample pts in voxel": None, "Sample pts in voxel arr (bx coord sys)": None, "All dose vals in voxel list": None, "Total num MC trials in voxel": None, "Arithmetic mean of dose in voxel": None, "Std dev of dose in voxel": None}
                for voxel_index in range(int(num_z_voxels)):
                    if voxel_index == 0 or voxel_index == range(int(num_z_voxels))[-1]:
                        voxel_z_end = voxel_z_begin + biopsy_z_voxel_length + extra_length_for_biopsy_end_cap_voxels
                    else:
                        voxel_z_end = voxel_z_begin + biopsy_z_voxel_length
                        
                    # find indices of the points in the biopsy that fall between the voxel bounds
                    sample_pts_indices_in_voxel_arr = np.asarray(np.logical_and(randomly_sampled_bx_pts_bx_coord_sys_arr[:,2] >= voxel_z_begin , randomly_sampled_bx_pts_bx_coord_sys_arr[:,2] <= voxel_z_end)).nonzero()
                    num_sample_pts_in_voxel = sample_pts_indices_in_voxel_arr[0].shape[0]
                    samples_pts_in_voxel_arr = np.take(randomly_sampled_bx_pts_bx_coord_sys_arr, sample_pts_indices_in_voxel_arr, axis=0)[0]
                    dose_vals_in_voxel_by_sampled_pt_index_arr = np.take(specific_bx_dose_results_list, sample_pts_indices_in_voxel_arr, axis = 0)[0]
                    dose_vals_in_voxel_list = dose_vals_in_voxel_by_sampled_pt_index_arr.flatten(order='C').tolist()
                    
                    total_num_MC_trials_in_voxel = num_sample_pts_in_voxel*num_simulations
                    if total_num_MC_trials_in_voxel < 1:
                        arithmetic_mean_dose_in_voxel = 'No data'
                    else:
                        arithmetic_mean_dose_in_voxel = statistics.mean(dose_vals_in_voxel_list)
                    if total_num_MC_trials_in_voxel <= 1:
                        std_dev_dose_in_voxel = 0
                    else:
                        std_dev_dose_in_voxel = statistics.stdev(dose_vals_in_voxel_list)

                    voxel_dose_dict = voxel_dose_dict_empty.copy()
                    voxel_dose_dict["Voxel z begin"] = voxel_z_begin
                    voxel_dose_dict["Voxel z end"] = voxel_z_end
                    voxel_dose_dict["Indices from all sample pts that are within voxel arr"] = sample_pts_indices_in_voxel_arr
                    voxel_dose_dict["Num sample pts in voxel"] = num_sample_pts_in_voxel
                    voxel_dose_dict["Sample pts in voxel arr (bx coord sys)"] = samples_pts_in_voxel_arr
                    voxel_dose_dict["Total num MC trials in voxel"] = total_num_MC_trials_in_voxel
                    voxel_dose_dict["All dose vals in voxel list"] = dose_vals_in_voxel_list
                    voxel_dose_dict["Arithmetic mean of dose in voxel"] = arithmetic_mean_dose_in_voxel
                    voxel_dose_dict["Std dev of dose in voxel"] = std_dev_dose_in_voxel
                    
                    voxelized_biopsy_dose_results_list[voxel_index] = voxel_dose_dict

                    voxel_z_begin = voxel_z_end
                
                # reorganize this data in a better way (didnt want to delete/change above code), but better to have a dictionary of lists rather than a list of dictionaries
                voxel_dose_dict_of_lists = voxel_dose_dict_empty.copy()
                #voxel_dict_of_lists = dict.fromkeys(voxel_dict_of_lists,[])
                for key,value in voxel_dose_dict_of_lists.items():
                    voxel_dose_dict_of_lists[key] = []
                voxel_dose_dict_of_lists["Voxel z range"] = []
                voxel_dose_dict_of_lists["Voxel z range rounded"] = []
                for voxel_index in range(int(num_z_voxels)):
                    voxel_dose_dict = voxelized_biopsy_dose_results_list[voxel_index]
                    voxel_dose_dict_of_lists["Voxel z begin"].append(voxel_dose_dict["Voxel z begin"])
                    voxel_dose_dict_of_lists["Voxel z end"].append(voxel_dose_dict["Voxel z end"])
                    voxel_dose_dict_of_lists["Voxel z range"].append([voxel_dose_dict["Voxel z begin"],voxel_dose_dict["Voxel z end"]])
                    voxel_dose_dict_of_lists["Voxel z range rounded"].append([round(voxel_dose_dict["Voxel z begin"],2),round(voxel_dose_dict["Voxel z end"],2)])
                    voxel_dose_dict_of_lists["Indices from all sample pts that are within voxel arr"].append(voxel_dose_dict["Indices from all sample pts that are within voxel arr"])
                    voxel_dose_dict_of_lists["Num sample pts in voxel"].append(voxel_dose_dict["Num sample pts in voxel"])
                    voxel_dose_dict_of_lists["Sample pts in voxel arr (bx coord sys)"].append(voxel_dose_dict["Sample pts in voxel arr (bx coord sys)"])
                    voxel_dose_dict_of_lists["Total num MC trials in voxel"].append(voxel_dose_dict["Total num MC trials in voxel"])
                    voxel_dose_dict_of_lists["All dose vals in voxel list"].append(voxel_dose_dict["All dose vals in voxel list"])
                    voxel_dose_dict_of_lists["Arithmetic mean of dose in voxel"].append(voxel_dose_dict["Arithmetic mean of dose in voxel"])
                    voxel_dose_dict_of_lists["Std dev of dose in voxel"].append(voxel_dose_dict["Std dev of dose in voxel"])
                    
                voxel_dose_dict_of_lists["Num voxels"] = int(num_z_voxels)
                voxelized_biopsy_dose_results_dict = voxel_dose_dict_of_lists
                
                specific_bx_structure["MC data: voxelized dose results list"] = voxelized_biopsy_dose_results_list
                specific_bx_structure["MC data: voxelized dose results dict (dict of lists)"] = voxelized_biopsy_dose_results_dict
                 
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, advance = 1)
            biopsies_progress.remove_task(biopsy_voxelize_each_bx_structure_dose_task)
            patients_progress.update(biopsy_voxelize_dose_task, advance = 1)
            completed_progress.update(biopsy_voxelize_dose_task_complete, advance = 1)
        patients_progress.update(biopsy_voxelize_dose_task, visible = False)
        completed_progress.update(biopsy_voxelize_dose_task_complete,visible = True)
        live_display.refresh()
        

        return master_structure_reference_dict


def normal_distribution_MLE_parallel(parallel_pool, data_2d_list):
    data_2d_arr = np.asarray(data_2d_list)
    num_rows = data_2d_arr.shape[0] # number of bx sampled pts
    args_list = [None]*num_rows
    for row_index, data_row in enumerate(data_2d_arr):
        args_list[row_index] = data_row # each row is a 1d array of dose values, each row should have length equal to number of MC simulations and the number of rows should be equal to the number of sampled BX points

    dosimetric_MLE_statistics_all_bx_pts_list = parallel_pool.map(normal_distribution_MLE,args_list) # each entry in the outer list corresponds to a bx point, each entry contains the following tuple: ((mu,se,var),(CI_lower,CI_upper))

    return dosimetric_MLE_statistics_all_bx_pts_list


def normal_distribution_MLE(data_1d_arr):
    estimators_mean_se_var = mf.normal_mean_se_var_estimatation(data_1d_arr)
    mu_estimator = estimators_mean_se_var[0]
    se_estimator = estimators_mean_se_var[1]
    mu_CI_estimation_tuple = mf.normal_CI_estimator(mu_estimator, se_estimator)
    stats_and_CI_tuple = (estimators_mean_se_var,mu_CI_estimation_tuple)
    return stats_and_CI_tuple
    



def dosimetric_localization_parallel(parallel_pool, bx_only_shifted_3darr, specific_bx_structure, dose_ref_dict, dose_ref, phys_space_dose_map_phys_coords_2d_arr, phys_space_dose_map_dose_2d_arr, num_dose_calc_NN):
    # build args list
    args_list = [None]*bx_only_shifted_3darr.shape[0]
    dose_ref_dict_roi = dose_ref_dict["Dose ID"]
    specific_bx_structure_roi = specific_bx_structure["ROI"]
    dose_data_KDtree = dose_ref_dict["KDtree"]
    for single_MC_trial_slice_index, bx_only_shifted_single_MC_trial_slice in enumerate(bx_only_shifted_3darr):
        single_MC_trial_arg = (dose_data_KDtree, bx_only_shifted_single_MC_trial_slice, specific_bx_structure_roi, dose_ref_dict_roi, dose_ref, phys_space_dose_map_phys_coords_2d_arr, phys_space_dose_map_dose_2d_arr, num_dose_calc_NN)
        args_list[single_MC_trial_slice_index] = single_MC_trial_arg
    
    # conduct the dosimetric localization in parallel. The MC trials are done in parallel.
    dosimetric_localiation_all_MC_trials_list = parallel_pool.starmap(dosimetric_localization_single_MC_trial, args_list)

    return dosimetric_localiation_all_MC_trials_list


def dosimetric_localization_single_MC_trial(dose_data_KDtree, bx_only_shifted_single_MC_trial_slice, specific_bx_structure_roi,dose_ref_dose_id,dose_ref,phys_space_dose_map_phys_coords_2d_arr, phys_space_dose_map_dose_2d_arr, num_dose_calc_NN):   
    nearest_neighbours_single_MC_trial_output = dose_data_KDtree.query(bx_only_shifted_single_MC_trial_slice, k=num_dose_calc_NN)
    nearest_neighbours_single_MC_trial_NN_parent_obj = nearest_neighbour_parent_dose(specific_bx_structure_roi,dose_ref_dose_id,dose_ref,phys_space_dose_map_phys_coords_2d_arr,bx_only_shifted_single_MC_trial_slice, phys_space_dose_map_dose_2d_arr, nearest_neighbours_single_MC_trial_output)
    return nearest_neighbours_single_MC_trial_NN_parent_obj

def calculate_binomial_containment_conf_intervals_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials):
    args_list = [(probability_estimator_list[j], num_trials, num_successes_list[j]) for j in range(len(probability_estimator_list))]
    confidence_interval_list = parallel_pool.starmap(mf.binomial_CI_estimator,args_list)
    return confidence_interval_list

def calculate_binomial_containment_stand_err_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials):
    args_list = [(probability_estimator_list[j], num_trials, num_successes_list[j]) for j in range(len(probability_estimator_list))]
    standard_err_list = parallel_pool.starmap(mf.binomial_se_estimator,args_list)
    return standard_err_list





def MC_simulator_shift_all_structures_generator_parallel(parallel_pool, patient_dict, structs_referenced_list, num_simulations):

    # build args list for parallel computing
    args_list = []
    patient_dict_updated_with_generated_samples = patient_dict.copy()
    for structure_type in structs_referenced_list:
        for specific_structure_index, specific_structure in enumerate(patient_dict[structure_type]):
            #spec_structure_zslice_wise_delaunay_obj_list = specific_structure["Delaunay triangulation zslice-wise list"]
            uncertainty_data_obj = specific_structure["Uncertainty data"]
            sp_struct_uncertainty_data_mean_arr = uncertainty_data_obj.uncertainty_data_mean_arr
            sp_struct_uncertainty_data_sigma_arr = uncertainty_data_obj.uncertainty_data_sigma_arr
            specific_structure_args = (structure_type,specific_structure_index,sp_struct_uncertainty_data_mean_arr,sp_struct_uncertainty_data_sigma_arr,num_simulations)
            args_list.append(specific_structure_args)

    
    # conduct random samples of all structure shifts in parallel
    sp_structure_normal_dist_shift_samples_and_structure_reference_list = parallel_pool.starmap(MC_simulator_shift_structure_generator,args_list)
    
    # update the patient dictionary
    for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list:
        structure_type = generated_shifts_info_list[0]
        specific_structure_index = generated_shifts_info_list[1]
        specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
        patient_dict_updated_with_generated_samples[structure_type][specific_structure_index]["MC data: Generated normal dist random samples arr"] = specific_structure_structure_normal_dist_shift_samples_arr

    return patient_dict_updated_with_generated_samples


def MC_simulator_shift_structure_generator(structure_type, specific_structure_index, sp_struct_uncertainty_data_mean_arr, sp_struct_uncertainty_data_sigma_arr, num_simulations):
    structure_normal_dist_shift_samples_arr = np.array([ 
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[0], scale=sp_struct_uncertainty_data_sigma_arr[0], size=num_simulations),  
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[1], scale=sp_struct_uncertainty_data_sigma_arr[1], size=num_simulations),  
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[2], scale=sp_struct_uncertainty_data_sigma_arr[2], size=num_simulations)],   
            dtype = float).T
    generated_shifts_info_list = [structure_type, specific_structure_index, structure_normal_dist_shift_samples_arr]
    return generated_shifts_info_list


def MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_parallel(parallel_pool, specific_bx_structure):
    randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
    randomly_sampled_bx_shifts_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]

    args_list = []
    for bx_shift_ind in range(randomly_sampled_bx_shifts_arr.shape[0]):
        arg = (randomly_sampled_bx_pts_arr, randomly_sampled_bx_shifts_arr[bx_shift_ind])
        args_list.append(arg)

    randomly_sampled_bx_pts_arr_bx_only_shift_arr_each_sampled_shift_list = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_bx_only_shift,args_list)

    num_bx_sampled_points = randomly_sampled_bx_pts_arr.shape[0]
    num_bx_sampled_shifts = randomly_sampled_bx_shifts_arr.shape[0]

    # create a 3d array that stores all the shifted bx data where each 3d slice is the shifted bx data for a fixed sampled bx shift, ie each slice is a sampled bx shift trial
    randomly_sampled_bx_pts_arr_bx_only_shift_3Darr = np.empty([num_bx_sampled_shifts,num_bx_sampled_points,3],dtype=float)

    # build the above described 3d array from the parallel results list
    for index,randomly_sampled_bx_pts_arr_bx_only_shift_arr in enumerate(randomly_sampled_bx_pts_arr_bx_only_shift_arr_each_sampled_shift_list):
        randomly_sampled_bx_pts_arr_bx_only_shift_3Darr[index] = randomly_sampled_bx_pts_arr_bx_only_shift_arr

    return randomly_sampled_bx_pts_arr_bx_only_shift_3Darr




def MC_simulator_translate_sampled_bx_points_arr_bx_only_shift(randomly_sampled_bx_pts_arr, bx_shift_vector):
    # For a single trial, all BX points are shifted by the same vector!
    randomly_sampled_bx_pts_arr_bx_only_shift_arr = randomly_sampled_bx_pts_arr + bx_shift_vector
    return randomly_sampled_bx_pts_arr_bx_only_shift_arr


def MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_parallel(parallel_pool, pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, blank_structure_shifted_bx_data_dict):
    # do each non bx structure sequentially
    structure_shifted_bx_data_dict = blank_structure_shifted_bx_data_dict.copy()
    for non_bx_struct_type in structs_referenced_list[1:]:
        for specific_non_bx_struct_index,specific_non_bx_struct in enumerate(pydicom_item[non_bx_struct_type]):
            # build args list for parallel computing
            specific_non_bx_struct_roi = specific_non_bx_struct["ROI"]
            specific_non_bx_struct_refnum = specific_non_bx_struct["Ref #"]

            args_list_non_bx_struct_shift = []
            num_2dslices_in_bx_3darr = bx_only_shifted_randomly_sampled_bx_pts_3Darr.shape[0]
            for slice_index in range(num_2dslices_in_bx_3darr): 
                bx_data_only_bx_shifted_2dslice_from_3d_arr = bx_only_shifted_randomly_sampled_bx_pts_3Darr[slice_index]
                non_bx_structure_shift_vector = specific_non_bx_struct["MC data: Generated normal dist random samples arr"][slice_index]
                bx_shift_vector = -non_bx_structure_shift_vector
                arg = (bx_data_only_bx_shifted_2dslice_from_3d_arr,bx_shift_vector)
                args_list_non_bx_struct_shift.append(arg)

            parallel_results_bx_shift_of_non_bx_structure_translation = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_structure_only_shift, args_list_non_bx_struct_shift)
            
            bx_data_both_non_bx_structure_shifted_and_bx_structure_shifted_3darr = np.asarray(parallel_results_bx_shift_of_non_bx_structure_translation)

            structure_shifted_bx_data_dict[specific_non_bx_struct_roi,non_bx_struct_type,specific_non_bx_struct_refnum,specific_non_bx_struct_index] = bx_data_both_non_bx_structure_shifted_and_bx_structure_shifted_3darr

    return structure_shifted_bx_data_dict


def MC_simulator_translate_sampled_bx_points_arr_structure_only_shift(randomly_sampled_bx_pts_arr, bx_shift_vector):
    # For a single trial, all BX points are shifted by the same vector!
    randomly_sampled_bx_pts_arr_struct_only_shift_arr = randomly_sampled_bx_pts_arr + bx_shift_vector
    return randomly_sampled_bx_pts_arr_struct_only_shift_arr



def point_containment_test_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, deulaunay_objs_zslice_wise_list, interslice_interpolation_information_of_containment_structure, test_pts_arr):
    num_pts = test_pts_arr.shape[0]
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud_zslice_delaunay = o3d.geometry.PointCloud()
    test_pts_point_cloud_zslice_delaunay.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    #st = time.time() # THIS IS THE SLOW SECTION!! 2 ORDERS OF MAGNTIUDE SLOWER THAN THE LOWER SECTION

    test_points_results_zslice_delaunay = point_containment_tools.test_global_convex_structure_containment_delaunay_parallel(parallel_pool, deulaunay_objs_zslice_wise_list, test_pts_list)
    for index,result in enumerate(test_points_results_zslice_delaunay):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud_zslice_delaunay.colors = o3d.utility.Vector3dVector(test_pt_colors)

    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (delaunay parallel):', elapsed_time, 'seconds')
    #print('___')
    #st = time.time()

    test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated = point_containment_tools.plane_point_in_polygon_concave(test_points_results_zslice_delaunay,interslice_interpolation_information_of_containment_structure, test_pts_point_cloud_zslice_delaunay)

    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (concave test):', elapsed_time, 'seconds')
    #print('___')


    return test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated



def point_containment_test_axis_aligned_bounding_box_and_zslice_wise_2d_PIP_parallel(parallel_pool, num_simulations, containment_structure_pts_arr, interslice_interpolation_information_of_containment_structure, test_pts_arr):
    num_pts = test_pts_arr.shape[0]
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud_after_axis_aligned_bounding_box_test = o3d.geometry.PointCloud()
    test_pts_point_cloud_after_axis_aligned_bounding_box_test.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    #st = time.time() 

    test_points_results_axis_aligned_bounding_box = point_containment_tools.test_global_axis_aligned_box_containment_parallel(parallel_pool, containment_structure_pts_arr, test_pts_list)
    for index,result in enumerate(test_points_results_axis_aligned_bounding_box):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud_after_axis_aligned_bounding_box_test.colors = o3d.utility.Vector3dVector(test_pt_colors)

    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (axis aligned bounding box):', elapsed_time, 'seconds')
    #print('___')
    #st = time.time()

    test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated = point_containment_tools.plane_point_in_polygon_concave(test_points_results_axis_aligned_bounding_box, interslice_interpolation_information_of_containment_structure, test_pts_point_cloud_after_axis_aligned_bounding_box_test)

    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (concave test):', elapsed_time, 'seconds')
    #print('___')

    # Lets plot everything to make sure everything is correct
    #test_pts_shifted_pcd = point_containment_tools.create_point_cloud(test_pts_arr)
    #plotting_funcs.plot_point_clouds(test_pts_shifted_pcd, label='Unknown')
    #plotting_funcs.plot_two_point_clouds_side_by_side(test_pts_point_cloud_after_axis_aligned_bounding_box_test, test_pts_point_cloud_concave_zslice_updated)



    return test_points_results_fully_concave, test_pts_point_cloud_concave_zslice_updated






def box_simulator_delaunay_zslice_wise_parallel(parallel_pool, num_simulations, deulaunay_objs_zslice_wise_list, point_cloud):
    # test points to test for inclusion
    num_pts = num_simulations
    max_bnd = point_cloud.get_max_bound()
    min_bnd = point_cloud.get_min_bound()
    center = point_cloud.get_center()
    if np.linalg.norm(max_bnd-center) >= np.linalg.norm(min_bnd-center): 
        largest_bnd = max_bnd
    else:
        largest_bnd = min_bnd
    bounding_box_size = np.linalg.norm(largest_bnd-center)
    test_pts = [np.random.uniform(-bounding_box_size,bounding_box_size, size = 3) for i in range(num_pts)]
    test_pts_arr = np.array(test_pts) + center
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud = o3d.geometry.PointCloud()
    test_pts_point_cloud.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    
    test_points_results = point_containment_tools.test_zslice_wise_containment_delaunay_parallel(parallel_pool, deulaunay_objs_zslice_wise_list, test_pts_list)
    for index,result in enumerate(test_points_results):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud.colors = o3d.utility.Vector3dVector(test_pt_colors)

    return test_points_results, test_pts_point_cloud


def box_simulator_delaunay_global_convex_structure_parallel(parallel_pool, num_simulations, deulaunay_obj, point_cloud):
    # test points to test for inclusion
    num_pts = num_simulations
    max_bnd = point_cloud.get_max_bound()
    min_bnd = point_cloud.get_min_bound()
    center = point_cloud.get_center()
    if np.linalg.norm(max_bnd-center) >= np.linalg.norm(min_bnd-center): 
        largest_bnd = max_bnd
    else:
        largest_bnd = min_bnd
    bounding_box_size = np.linalg.norm(largest_bnd-center)
    test_pts = [np.random.uniform(-bounding_box_size,bounding_box_size, size = 3) for i in range(num_pts)]
    test_pts_arr = np.array(test_pts) + center
    test_pts_list = test_pts_arr.tolist()
    test_pts_point_cloud = o3d.geometry.PointCloud()
    test_pts_point_cloud.points = o3d.utility.Vector3dVector(test_pts_arr)
    test_pt_colors = np.empty([num_pts,3], dtype=float)
    
    
    test_points_results = point_containment_tools.test_global_convex_structure_containment_delaunay_parallel(parallel_pool, deulaunay_obj, test_pts_list)
    for index,result in enumerate(test_points_results):
        test_pt_colors[index] = result[2]
    test_pts_point_cloud.colors = o3d.utility.Vector3dVector(test_pt_colors)

    return test_points_results, test_pts_point_cloud





def point_sampler_from_global_delaunay_convex_structure_for_sequential(num_samples, delaunay_global_convex_structure_tri, reconstructed_bx_point_cloud):
    insert_index = 0
    reconstructed_bx_point_cloud_color = np.array([0,0,1])
    reconstructed_bx_point_cloud.paint_uniform_color(reconstructed_bx_point_cloud_color)

    bx_samples_arr = np.empty((num_samples,3),dtype=float)
    axis_aligned_bounding_box = reconstructed_bx_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    
    while insert_index < num_samples:
        x_val = np.random.uniform(min_bounds[0], max_bounds[0])
        y_val = np.random.uniform(min_bounds[1], max_bounds[1])
        z_val = np.random.uniform(min_bounds[2], max_bounds[2])
        random_point_within_bounding_box = np.array([x_val,y_val,z_val],dtype=float)
        
        containment_result_bool = point_containment_tools.convex_bx_structure_global_test_point_containment(delaunay_global_convex_structure_tri,random_point_within_bounding_box)
        
        
        random_point_pcd = o3d.geometry.PointCloud()
        random_point_pcd.points = o3d.utility.Vector3dVector(np.array([random_point_within_bounding_box]))
        random_point_pcd_color = np.array([0,1,0])
        random_point_pcd.paint_uniform_color(random_point_pcd_color)
        #plotting_funcs.plot_geometries(reconstructed_bx_point_cloud,random_point_pcd)
        #print(containment_result_bool)
        if containment_result_bool == True:
            bx_samples_arr[insert_index] = random_point_within_bounding_box
            insert_index = insert_index + 1
        else:
            pass
    
    bx_samples_arr_point_cloud_color = np.random.uniform(0, 0.7, size=3)
    bx_samples_arr_point_cloud = point_containment_tools.create_point_cloud(bx_samples_arr, bx_samples_arr_point_cloud_color)
    
    return bx_samples_arr, bx_samples_arr_point_cloud, axis_aligned_bounding_box


def point_sampler_from_global_delaunay_convex_structure_parallel(num_samples, delaunay_global_convex_structure_tri, reconstructed_bx_arr, patientUID, structure_type, specific_structure_index):
    insert_index = 0
    reconstructed_bx_point_cloud = point_containment_tools.create_point_cloud(reconstructed_bx_arr)
    reconstructed_bx_point_cloud_color = np.array([0,0,1])
    reconstructed_bx_point_cloud.paint_uniform_color(reconstructed_bx_point_cloud_color)

    bx_samples_arr = np.empty((num_samples,3),dtype=float)
    axis_aligned_bounding_box = reconstructed_bx_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    
    while insert_index < num_samples:
        x_val = np.random.uniform(min_bounds[0], max_bounds[0])
        y_val = np.random.uniform(min_bounds[1], max_bounds[1])
        z_val = np.random.uniform(min_bounds[2], max_bounds[2])
        random_point_within_bounding_box = np.array([x_val,y_val,z_val],dtype=float)
        
        containment_result_bool = point_containment_tools.convex_bx_structure_global_test_point_containment(delaunay_global_convex_structure_tri,random_point_within_bounding_box)
        
        
        random_point_pcd = o3d.geometry.PointCloud()
        random_point_pcd.points = o3d.utility.Vector3dVector(np.array([random_point_within_bounding_box]))
        random_point_pcd_color = np.array([0,1,0])
        random_point_pcd.paint_uniform_color(random_point_pcd_color)
        #plotting_funcs.plot_geometries(reconstructed_bx_point_cloud,random_point_pcd)
        #print(containment_result_bool)
        if containment_result_bool == True:
            bx_samples_arr[insert_index] = random_point_within_bounding_box
            insert_index = insert_index + 1
        else:
            pass
        
    
    bx_samples_arr_point_cloud_color = np.random.uniform(0, 0.7, size=3)
    bx_samples_arr_point_cloud = point_containment_tools.create_point_cloud(bx_samples_arr, bx_samples_arr_point_cloud_color)
    
    return bx_samples_arr, axis_aligned_bounding_box_points_arr, {"Patient UID": patientUID, "Structure type": structure_type, "Specific structure index": specific_structure_index}


class nearest_neighbour_parent_dose:
    def __init__(self,BX_struct_name,comparison_struct_name,comparison_struct_type,comparison_structure_points_that_made_KDtree,queried_BX_points, dose_2d_arr, NN_search_output):
        self.BX_structure_name = BX_struct_name
        self.comparison_structure_name = comparison_struct_name
        self.comparison_structure_type = comparison_struct_type
        #self.comparison_structure_points = comparison_structure_points_that_made_KDtree
        self.queried_Bx_points = queried_BX_points
        self.NN_search_output = NN_search_output
        self.dose_arr = dose_2d_arr
        self.NN_data_list = self.NN_list_builder(comparison_structure_points_that_made_KDtree)
        

    def NN_list_builder(self,comparison_structure_points_that_made_KDtree):
        comparison_structure_NN_distances = self.NN_search_output[0]
        comparison_structure_NN_indices = self.NN_search_output[1]
        nearest_points_on_comparison_struct = comparison_structure_points_that_made_KDtree[comparison_structure_NN_indices]
        nearest_doses_list = self.dose_arr[comparison_structure_NN_indices]
        nearest_doses_weighted_mean_list = np.average(nearest_doses_list, axis=1,weights = comparison_structure_NN_distances).tolist()
        NN_data_list = [nearest_neighbour_child_dose(self.queried_Bx_points[index], nearest_points_on_comparison_struct[index], comparison_structure_NN_distances[index], nearest_doses_weighted_mean_list[index]) for index in range(0,len(self.queried_Bx_points))]
        #NN_data_list = [{"Queried BX pt": self.queried_Bx_points[index], "NN pt on comparison struct": nearest_points_on_comparison_struct[index], "Euclidean distance": comparison_structure_NN_distances[index]} for index in range(0,len(self.queried_Bx_points))]
        return NN_data_list


class nearest_neighbour_child_dose:
    def __init__(self, queried_BX_pt, NN_pt_on_comparison_struct, euclidean_dist, nearest_dose):
        self.queried_BX_pt = queried_BX_pt
        self.NN_pt_on_comparison_struct = NN_pt_on_comparison_struct
        self.euclidean_dist = euclidean_dist
        self.nearest_dose = nearest_dose