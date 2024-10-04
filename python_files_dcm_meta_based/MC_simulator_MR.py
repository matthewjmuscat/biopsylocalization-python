import scipy
import numpy as np
import lattice_reconstruction_tools
import misc_tools
import cupy as cp
import MC_simulator_convex
import mr_localizers
import copy
import plotting_funcs
import point_containment_tools

def simulator_parallel(live_display,
                       layout_groups,
                       master_structure_reference_dict,
                       master_structure_info_dict,
                       structs_referenced_list,
                       mr_adc_ref,
                       bx_ref,
                       num_mr_calc_NN,
                       idw_power,
                       raw_data_mc_MR_dump_bool,
                       show_NN_mr_adc_demonstration_plots,
                       stopwatch,
                       dose_views_jsons_paths_list,
                       perform_mc_mr_sim,
                       show_NN_mr_adc_demonstration_plots_all_trials_at_once):
    
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    with live_display:
        live_display.start(refresh = True)
        num_patients = master_structure_info_dict["Global"]["Num cases"]
        num_MC_mr_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC MR simulations"]
        bx_sample_pt_lattice_spacing = master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing (mm)"]
        bx_sample_pts_volume_element = master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"]

        


        #live_display.stop()
        ### MR ADC LOCALIZATION!
        calc_MR_NN_biopsy_task = patients_progress.add_task("[red]Calculating NN MR localization  [{}]...".format("initializing"), total=num_patients)
        calc_MR_NN_biopsy_task_complete = completed_progress.add_task("[green]Calculating NN MR localization ", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_MR_NN_biopsy_task, description = "[red]Calculating NN MR localization [{}]...".format(patientUID))

            ### THIS LINE IS ALWAYS REQUIRED AS IT IS NOT NECESSARY FOR A PATIENT TO HAVE AN MR IMAGE TO RUN THE PROGRAMME!
            if mr_adc_ref not in pydicom_item:
                patients_progress.update(calc_MR_NN_biopsy_task, advance = 1)
                completed_progress.update(calc_MR_NN_biopsy_task_complete, advance = 1)
                continue

            mr_adc_subdict = pydicom_item[mr_adc_ref]
            

            filtered_non_negative_adc_mr_phys_space_arr = lattice_reconstruction_tools.reconstruct_mr_lattice_with_coordinates_from_dict_v2(mr_adc_subdict, filter_out_negatives = True)

            # create KDtree for MR ADC data

            mr_adc_phys_coords_arr = filtered_non_negative_adc_mr_phys_space_arr[:,0:3]
            mr_adc_vals_arr = filtered_non_negative_adc_mr_phys_space_arr[:,3]

            mr_adc_KDtree = scipy.spatial.KDTree(mr_adc_phys_coords_arr)
            mr_adc_subdict["KDtree"] = mr_adc_KDtree
            

            # code for the plotting of the below NN search of sampled bx pts
            lattice_mr_adc_pcd = mr_adc_subdict["MR ADC grid point cloud"]
            thresholded_lattice_mr_adc_pcd = mr_adc_subdict["MR ADC grid point cloud thresholded"]

            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            mr_adc_calc_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(mr_adc_calc_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                


                bx_structure_info_dict = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_bx_structure)
                

                # Replaced below code with a function
                bx_only_shifted_stacked_2darr = MC_simulator_convex.prepare_2d_stacked_arr_biopsy_only_shifted_with_nominal(specific_bx_structure,
                                            num_MC_mr_simulations)
                """
                bx_only_shifted_3darr = cp.asnumpy(specific_bx_structure["MC data: bx only shifted 3darr"]) # note that the 3rd dimension slices are each MC trial
                bx_only_shifted_3darr_cutoff = bx_only_shifted_3darr[0:num_MC_dose_simulations]
                unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                unshifted_bx_sampled_pts_arr_3darr = np.expand_dims(unshifted_bx_sampled_pts_arr, axis=0)
                nominal_and_bx_only_shifted_3darr = np.concatenate((unshifted_bx_sampled_pts_arr_3darr,bx_only_shifted_3darr_cutoff))
                bx_only_shifted_stacked_2darr = np.reshape(nominal_and_bx_only_shifted_3darr, (-1,3) , order = 'C')
                """

                

                
                mr_adc_calc_task = indeterminate_progress_sub.add_task("[cyan]~~Conducting NN search [{}]...".format(specific_bx_structure_roi), total = None)

                ### THIS DATAFRAME CONSUMES TOO MUCH MEMORY TO CARRY IT THROUGHOUT THE PROGRAMME, NEED TO PARSE IMMEDIATELY,
                ### CAN CONSIDER SAVING TO DISK... ONE STRATEGY COULD BE TO CONTINUALLY APPEND TO A CSV ON DISK!
                mr_nearest_neighbour_results_dataframe = mr_localizers.mr_localization_dataframe_version(bx_only_shifted_stacked_2darr,
                                            patientUID, 
                                            bx_structure_info_dict, 
                                            mr_adc_KDtree, 
                                            mr_adc_vals_arr, 
                                            num_mr_calc_NN,
                                            num_MC_mr_simulations,
                                            num_sample_pts_per_bx,
                                            idw_power)
                
                
                # TAKE A DUMP?
                if raw_data_mc_MR_dump_bool == True:
                    raw_mc_output_dir = master_structure_info_dict["Global"]["Raw MC output dir"]
                    mr_adc_raw_results_csv_name = str(patientUID)+"-"+str(specific_bx_structure_roi)+"-"'mc_raw_results_MR_ADC.csv'
                    mr_adc_raw_results_csv = raw_mc_output_dir.joinpath(mr_adc_raw_results_csv_name)
                    with open(mr_adc_raw_results_csv, 'a') as temp_file_obj:
                        mr_nearest_neighbour_results_dataframe.to_csv(temp_file_obj, mode='a', index=False, header=temp_file_obj.tell()==0)

                indeterminate_progress_sub.remove_task(mr_adc_calc_task)

                # plot everything to make sure its working properly!
                if show_NN_mr_adc_demonstration_plots == True:
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                    for trial_num in np.arange(0,num_MC_mr_simulations+1):
                        NN_pts_on_comparison_struct_for_all_points_concatenated = np.concatenate(mr_nearest_neighbour_results_dataframe[mr_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Nearest phys space points"].to_numpy())
                        NN_mr_vals_on_comparison_struct_for_all_points_concatenated = np.concatenate(mr_nearest_neighbour_results_dataframe[mr_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Nearest MR vals"].to_numpy())
                        queried_bx_pts_arr_concatenated = np.stack(mr_nearest_neighbour_results_dataframe[mr_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Struct test pt vec"].to_numpy())
                        queried_bx_pts_assigned_mr_vals_arr_concatenated = mr_nearest_neighbour_results_dataframe[mr_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["MR val (interpolated)"].to_numpy()
                        
                        patients_progress.stop_task(calc_MR_NN_biopsy_task)
                        completed_progress.stop_task(calc_MR_NN_biopsy_task_complete)
                        stopwatch.stop()
                        #plotting_funcs.dose_point_cloud_with_dose_labels_for_animation(NN_pts_on_comparison_struct_for_all_points_concatenated, NN_doses_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, queried_bx_pts_assigned_doses_arr_concatenated, num_dose_calc_NN, draw_lines = True)
                        geometry_list_full_mr_adc_lattice = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, lattice_mr_adc_pcd, NN_pts_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, num_mr_calc_NN, draw_lines = True)
                        geometry_list_thresholded_mr_adc_lattice = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, thresholded_lattice_mr_adc_pcd, NN_pts_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, num_mr_calc_NN, draw_lines = True)
                        plotting_funcs.plot_two_views_side_by_side(geometry_list_thresholded_mr_adc_lattice, dose_views_jsons_paths_list[0], geometry_list_thresholded_mr_adc_lattice, dose_views_jsons_paths_list[1])
                        plotting_funcs.plot_two_views_side_by_side(geometry_list_thresholded_mr_adc_lattice, dose_views_jsons_paths_list[2], geometry_list_thresholded_mr_adc_lattice, dose_views_jsons_paths_list[3])
                        plotting_funcs.dose_point_cloud_with_dose_labels_for_animation_plotly(NN_pts_on_comparison_struct_for_all_points_concatenated, NN_mr_vals_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, queried_bx_pts_assigned_mr_vals_arr_concatenated, num_mr_calc_NN, aspect_mode_input = 'data', draw_lines = False, axes_visible=True)
                        plotting_funcs.dose_point_cloud_with_dose_labels_for_animation_plotly(NN_pts_on_comparison_struct_for_all_points_concatenated, NN_mr_vals_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, queried_bx_pts_assigned_mr_vals_arr_concatenated, num_mr_calc_NN, aspect_mode_input = 'data', draw_lines = True, axes_visible=True)
                        stopwatch.start()
                        patients_progress.start_task(calc_MR_NN_biopsy_task)
                        completed_progress.start_task(calc_MR_NN_biopsy_task_complete)
                    
                    del geometry_list_full_mr_adc_lattice
                    del geometry_list_thresholded_mr_adc_lattice

                else:
                    pass

                if show_NN_mr_adc_demonstration_plots_all_trials_at_once == True:
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))

                    NN_pts_on_comparison_struct_for_all_points_concatenated = np.concatenate(mr_nearest_neighbour_results_dataframe["Nearest phys space points"].to_numpy())
                    queried_bx_pts_arr_concatenated = np.stack(mr_nearest_neighbour_results_dataframe["Struct test pt vec"].to_numpy())
                    
                    NN_mr_adc_locations_pointcloud = point_containment_tools.create_point_cloud(NN_pts_on_comparison_struct_for_all_points_concatenated)
                    queried_bx_pts_locations_pointcloud = point_containment_tools.create_point_cloud(queried_bx_pts_arr_concatenated, color = np.array([0,1,0]))


                    pcd_list = [unshifted_bx_sampled_pts_copy_pcd, thresholded_lattice_mr_adc_pcd, NN_mr_adc_locations_pointcloud, queried_bx_pts_locations_pointcloud]
                    
                    stopwatch.stop()
                    plotting_funcs.plot_geometries(*pcd_list)
                    stopwatch.start()

                    # Now include background anatomy
                    for other_struct_type in structs_referenced_list:
                        if other_struct_type == bx_ref:
                                continue
                        for specific_structure_index, specific_structure in enumerate(pydicom_item[other_struct_type]):
                            structure_pcd = specific_structure["Interpolated structure point cloud dict"]["Full"]
                            pcd_list.append(structure_pcd)
                    
                    stopwatch.stop()
                    plotting_funcs.plot_geometries(*pcd_list)
                    stopwatch.start()


                    #stopwatch.stop()
                    #live_display.stop()
                    #geometry_list_full_mr_adc_lattice = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, lattice_mr_adc_pcd, NN_pts_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, num_mr_calc_NN, draw_lines = True)
                    #geometry_list_thresholded_mr_adc_lattice = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, thresholded_lattice_mr_adc_pcd, NN_pts_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, num_mr_calc_NN, draw_lines = True)
                    #stopwatch.start()

                    del pcd_list
                else:
                    pass
                

                # Cant save these dataframes, they take up too much memory! Need to parse data right away
                #specific_bx_structure['MC data: bx to MR ADC NN search results dataframe'] = mr_nearest_neighbour_results_dataframe # Note that trial 0 is the nominal position


                ### COMPILE DATA STRAIGHT AWAY!
                mr_adc_nearest_neighbour_results_dataframe_pivoted = mr_nearest_neighbour_results_dataframe.pivot(index = "Original pt index", columns="Trial num", values = "MR val (interpolated)")
                del mr_nearest_neighbour_results_dataframe
                
                # It seems pivoting already sorts the indices and columns, but just to be sure I do it manually anyways
                mr_adc_nearest_neighbour_results_dataframe_pivoted_ensured_sorted = mr_adc_nearest_neighbour_results_dataframe_pivoted.sort_index(axis = 0).sort_index(axis = 1)
                del mr_adc_nearest_neighbour_results_dataframe_pivoted
                
                # Note that each row is a specific biopsy point, while the column is a particular MC trial
                mr_adc_localization_mr_vals_by_bx_point_nominal_and_all_trials_arr = mr_adc_nearest_neighbour_results_dataframe_pivoted_ensured_sorted.to_numpy()
                del mr_adc_nearest_neighbour_results_dataframe_pivoted_ensured_sorted


                # Update master dictionary
                # Nominal and MC trials
                specific_bx_structure["MC data: MR ADC vals for each sampled bx pt arr (nominal & all MC trials)"] = mr_adc_localization_mr_vals_by_bx_point_nominal_and_all_trials_arr
                
                del mr_adc_localization_mr_vals_by_bx_point_nominal_and_all_trials_arr


                biopsies_progress.update(mr_adc_calc_biopsy_task, advance=1)
            biopsies_progress.remove_task(mr_adc_calc_biopsy_task)
            patients_progress.update(calc_MR_NN_biopsy_task, advance = 1)
            completed_progress.update(calc_MR_NN_biopsy_task_complete, advance = 1)
        patients_progress.update(calc_MR_NN_biopsy_task, visible = False)
        completed_progress.update(calc_MR_NN_biopsy_task_complete, visible = True)
        live_display.refresh()



    master_structure_info_dict['Global']["MC info"]['MC MR sim performed'] = perform_mc_mr_sim


    return master_structure_reference_dict, live_display