import cupy_functions
import cupy as cp
import MC_simulator_convex
import copy
import numpy as np
from shapely import Polygon
import cuspatial
import geopandas
import point_containment_tools
import plotting_funcs
import cudf
import fanova_mathfuncs
import scipy

def fanova_analysis(
            parallel_pool, 
            live_display,
            stopwatch, 
            layout_groups, 
            master_structure_reference_dict, 
            master_structure_info_dict,
            structs_referenced_list,
            bx_ref,
            dose_ref,
            biopsy_needle_compartment_length,
            simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
            num_FANOVA_containment_simulations_input,
            num_FANOVA_dose_simulations_input,
            fanova_plot_uniform_shifts_to_check_plotly,
            show_fanova_containment_demonstration_plots,
            plot_cupy_fanova_containment_distribution_results,
            num_sobol_bootstraps,
            sobol_indices_bootstrap_conf_interval,
            show_NN_FANOVA_dose_demonstration_plots,
            num_dose_calc_NN,
            dose_views_jsons_paths_list,
            perform_dose_fanova,
            perform_containment_fanova
            ):
    
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    with live_display:
        live_display.start(refresh = True)

        num_patients = master_structure_info_dict["Global"]["Num patients"]
        num_global_structures = master_structure_info_dict["Global"]["Num structures"]
        bx_sample_pt_lattice_spacing = master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing"]


        max_fanova_simulations = max(num_FANOVA_dose_simulations_input,num_FANOVA_containment_simulations_input)

        #live_display.stop()
        default_output = "Initializing"
        processing_patients_task_main_description = "[red]Generating {} FANOVA samples for {} structures [{}]...".format(max_fanova_simulations,num_global_structures,default_output)
        processing_patients_task_completed_main_description = "[green]Generating {} FANOVA samples for {} structures".format(max_fanova_simulations,num_global_structures)
        processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total = num_patients)
        processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total = num_patients, visible = False)

        # simulate all structure shifts in parallel and update the master reference dict
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            processing_patients_task_main_description = "[red]Generating {} MC samples for {} structures [{}]...".format(max_fanova_simulations,num_global_structures,patientUID)
            patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
            if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
                sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list_A = cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(pydicom_item, bx_ref, biopsy_needle_compartment_length, max_fanova_simulations)
                sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list_B = cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(pydicom_item, bx_ref, biopsy_needle_compartment_length, max_fanova_simulations)
                
                # update the patient dictionary
                for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list_A:
                    structure_type = generated_shifts_info_list[0]
                    specific_structure_index = generated_shifts_info_list[1]
                    specific_structure_structure_uniform_dist_shift_samples_arr = generated_shifts_info_list[2]
                    pydicom_item[structure_type][specific_structure_index]["FANOVA: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr (A)"] = specific_structure_structure_uniform_dist_shift_samples_arr
                # update the patient dictionary
                for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list_B:
                    structure_type = generated_shifts_info_list[0]
                    specific_structure_index = generated_shifts_info_list[1]
                    specific_structure_structure_uniform_dist_shift_samples_arr = generated_shifts_info_list[2]
                    pydicom_item[structure_type][specific_structure_index]["FANOVA: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr (B)"] = specific_structure_structure_uniform_dist_shift_samples_arr



            sp_structure_normal_dist_shift_samples_and_structure_reference_list_A = cupy_functions.MC_simulator_shift_all_structures_generator_cupy(pydicom_item, structs_referenced_list, max_fanova_simulations)
            sp_structure_normal_dist_shift_samples_and_structure_reference_list_B = cupy_functions.MC_simulator_shift_all_structures_generator_cupy(pydicom_item, structs_referenced_list, max_fanova_simulations)

            # update the patient dictionary
            for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list_A:
                structure_type = generated_shifts_info_list[0]
                specific_structure_index = generated_shifts_info_list[1]
                specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
                pydicom_item[structure_type][specific_structure_index]["FANOVA: Generated normal dist random samples arr (A)"] = specific_structure_structure_normal_dist_shift_samples_arr
            # update the patient dictionary
            for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list_B:
                structure_type = generated_shifts_info_list[0]
                specific_structure_index = generated_shifts_info_list[1]
                specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
                pydicom_item[structure_type][specific_structure_index]["FANOVA: Generated normal dist random samples arr (B)"] = specific_structure_structure_normal_dist_shift_samples_arr
            


            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_task_completed, advance = 1)
        patients_progress.update(processing_patients_task, visible = False, refresh = True)
        completed_progress.update(processing_patients_task_completed, visible = True, refresh = True)
        live_display.refresh()


        default_output = "Initializing"
        processing_patients_task_main_description = "[red]Transforming FANOVA biopsies [{}]...".format(default_output)
        processing_patients_task_completed_main_description = "[green]Transforming FANOVA biopsies"
        processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total = num_patients)
        processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total = num_patients, visible = False)

        # simulate all structure shifts in parallel and update the master reference dict
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            processing_patients_task_main_description = "[red]Transforming FANOVA biopsies [{}]...".format(patientUID)
            patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                
                specific_structure_structure_normal_dist_shift_samples_arr_A = specific_bx_structure["FANOVA: Generated normal dist random samples arr (A)"]
                specific_structure_structure_normal_dist_shift_samples_arr_B = specific_bx_structure["FANOVA: Generated normal dist random samples arr (B)"]
                
                if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
                    specific_structure_structure_uniform_dist_shift_samples_arr_A = specific_bx_structure["FANOVA: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr (A)"]
                    specific_structure_structure_uniform_dist_shift_samples_arr_B = specific_bx_structure["FANOVA: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr (B)"]
                    
                    fanova_bx_samples_A = cp.empty((specific_structure_structure_normal_dist_shift_samples_arr_A.shape[0],specific_structure_structure_normal_dist_shift_samples_arr_A.shape[1]+1))
                    fanova_bx_samples_A[:,0:3] = specific_structure_structure_normal_dist_shift_samples_arr_A
                    fanova_bx_samples_A[:,3] = specific_structure_structure_uniform_dist_shift_samples_arr_A

                    fanova_bx_samples_B = cp.empty((specific_structure_structure_normal_dist_shift_samples_arr_B.shape[0],specific_structure_structure_normal_dist_shift_samples_arr_B.shape[1]+1))
                    fanova_bx_samples_B[:,0:3] = specific_structure_structure_normal_dist_shift_samples_arr_B
                    fanova_bx_samples_B[:,3] = specific_structure_structure_uniform_dist_shift_samples_arr_B

                else:
                    fanova_bx_samples_A = specific_structure_structure_normal_dist_shift_samples_arr_A
                    fanova_bx_samples_B = specific_structure_structure_normal_dist_shift_samples_arr_B

                num_variance_vars = fanova_bx_samples_A.shape[1]
                fanova_bx_samples_j_matrices_3d_arr = cp.empty((num_variance_vars,fanova_bx_samples_A.shape[0],fanova_bx_samples_A.shape[1]))
                for j in range(0,fanova_bx_samples_A.shape[1]):
                    fanova_bx_samples_j = copy.deepcopy(fanova_bx_samples_A)
                    fanova_bx_samples_j[:,j] = fanova_bx_samples_B[:,j]
                    fanova_bx_samples_j_matrices_3d_arr[j,:,:] = fanova_bx_samples_j

                specific_bx_structure["FANOVA: permuted sample matrices 3d arr"] = cp.asnumpy(fanova_bx_samples_j_matrices_3d_arr)
                specific_bx_structure["FANOVA: sample matrix A"] = cp.asnumpy(fanova_bx_samples_A)
                specific_bx_structure["FANOVA: sample matrix B"] = cp.asnumpy(fanova_bx_samples_B)
                del fanova_bx_samples_j_matrices_3d_arr
                del fanova_bx_samples_A
                del fanova_bx_samples_B

            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_task_completed, advance = 1)
        patients_progress.update(processing_patients_task, visible = False, refresh = True)
        completed_progress.update(processing_patients_task_completed, visible = True, refresh = True)
        live_display.refresh()
        
        # Save the number of FANOVA variance vars
        master_structure_info_dict["Global"]["FANOVA: num variance vars"] = num_variance_vars

        default_output = "Initializing"
        processing_patients_task_main_description = "[red]Generating FANOVA arrays [{}]...".format(default_output)
        processing_patients_task_completed_main_description = "[green]Generating FANOVA arrays"
        processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total = num_patients)
        processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total = num_patients, visible = False)

        # simulate all structure shifts in parallel and update the master reference dict
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            processing_patients_task_main_description = "[red]Generating FANOVA arrays [{}]...".format(patientUID)
            patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sampled_sp_bx_pts = specific_bx_structure["Num sampled bx pts"]

                specific_bx_structure["FANOVA: Generated uniform (biopsy needle compartment) random vectors samples arr dict"] = {}
                specific_bx_structure["FANOVA: Total rigid shift vectors arr dict"] = {}
                specific_bx_structure["FANOVA: bx only shifted 3darr dict"] = {}

                fanova_bx_samples_A = cp.array(specific_bx_structure["FANOVA: sample matrix A"])
                matrix_key = 'A'
                randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr = cupy_functions.fanova_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, 
                                                                                         simulate_uniform_bx_shifts_due_to_bx_needle_compartment, 
                                                                                         fanova_plot_uniform_shifts_to_check_plotly, 
                                                                                         num_sampled_sp_bx_pts, 
                                                                                         max_fanova_simulations, 
                                                                                         fanova_bx_samples_A,
                                                                                         matrix_key)
                specific_bx_structure["FANOVA: bx only shifted 3darr dict"][matrix_key] = cp.asnumpy(randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr)
                del fanova_bx_samples_A
                del randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr
                fanova_bx_samples_B = cp.array(specific_bx_structure["FANOVA: sample matrix B"])
                matrix_key = 'B'
                randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr = cupy_functions.fanova_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, 
                                                                                         simulate_uniform_bx_shifts_due_to_bx_needle_compartment, 
                                                                                         fanova_plot_uniform_shifts_to_check_plotly, 
                                                                                         num_sampled_sp_bx_pts, 
                                                                                         max_fanova_simulations, 
                                                                                         fanova_bx_samples_B,
                                                                                         matrix_key)
                specific_bx_structure["FANOVA: bx only shifted 3darr dict"][matrix_key] = cp.asnumpy(randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr)
                del fanova_bx_samples_B
                del randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr

                fanova_bx_samples_j_matrices_3d_arr = cp.array(specific_bx_structure["FANOVA: permuted sample matrices 3d arr"])

                for j in range(0,fanova_bx_samples_j_matrices_3d_arr.shape[0]):
                    matrix_key = str(j)
                    fanova_bx_samples_j = fanova_bx_samples_j_matrices_3d_arr[j,:,:]
                    randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr = cupy_functions.fanova_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, 
                                                                                         simulate_uniform_bx_shifts_due_to_bx_needle_compartment, 
                                                                                         fanova_plot_uniform_shifts_to_check_plotly, 
                                                                                         num_sampled_sp_bx_pts, 
                                                                                         max_fanova_simulations, 
                                                                                         fanova_bx_samples_j,
                                                                                         matrix_key)
                    specific_bx_structure["FANOVA: bx only shifted 3darr dict"][matrix_key] = cp.asnumpy(randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr)
                    del randomly_sampled_bx_pts_arr_bx_only_shift_cp_3Darr
                del fanova_bx_samples_j_matrices_3d_arr

            patients_progress.update(processing_patients_task, advance=1)
            completed_progress.update(processing_patients_task_completed, advance=1)
        patients_progress.update(processing_patients_task, visible=False)
        completed_progress.update(processing_patients_task_completed, visible=True)
        live_display.refresh()


        #live_display.stop()
        if perform_containment_fanova == True:
            testing_biopsy_containment_patient_task = patients_progress.add_task("[red]FANOVA: Testing biopsy containment (cuspatial)...", total=num_patients)
            testing_biopsy_containment_patient_task_completed = completed_progress.add_task("[green]FANOVA: Testing biopsy containment (cuspatial)", total=num_patients, visible = False)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

                patients_progress.update(testing_biopsy_containment_patient_task, description = "[red]Testing biopsy containment (cuspatial) [{}]...".format(patientUID))
                testing_biopsy_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)           
                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    bx_specific_biopsy_containment_desc = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi)
                    biopsies_progress.update(testing_biopsy_containment_task, description = bx_specific_biopsy_containment_desc)

                    # For nominal position containment test
                    unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                    
                    # paint the unshifted bx sampled points purple for later viewing
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                    
                    # extract fanova 3d array samples
                    shifted_bx_data_dict = specific_bx_structure["FANOVA: bx only shifted 3darr dict"]

                    testing_each_fanova_matrix_containment_task = structures_progress.add_task("[cyan]~~For each FANOVA matrix [{}]...".format("initializing"), total=len(shifted_bx_data_dict))
                    
                    #structure_shifted_bx_data_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx and structure shifted dict"] 
                    
                    relative_structure_containment_results_data_frames_list = []
                    for matrix_key,shifted_bx_data_3darr_cp in shifted_bx_data_dict.items():
                        
                        shifted_bx_data_3darr = cp.asnumpy(shifted_bx_data_3darr_cp)
                        structures_progress.update(testing_each_fanova_matrix_containment_task, description = "[cyan]~~For each FANOVA matrix [{}]...".format(matrix_key))
                        for non_bx_struct_type in structs_referenced_list[1:]:
                            for specific_non_bx_structure_index, specific_non_bx_structure in enumerate(pydicom_item[non_bx_struct_type]):
                                # Extract and calcualte relative structure info
                                non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_struct_type][specific_non_bx_structure_index]["Inter-slice interpolation information"]
                                non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                                interpolated_zvlas_list = non_bx_struct_interslice_interpolation_information.zslice_vals_after_interpolation_list
                                non_bx_struct_zslices_list = non_bx_struct_interslice_interpolation_information.interpolated_pts_list
                                non_bx_struct_max_zval = max(interpolated_zvlas_list)
                                non_bx_struct_min_zval = min(interpolated_zvlas_list)
                                non_bx_struct_zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in non_bx_struct_zslices_list]
                                non_bx_struct_zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(non_bx_struct_zslices_polygons_list))

                                # Point clouds
                                non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))
                                prostate_interslice_interpolation_information = master_structure_reference_dict[patientUID]['OAR ref'][0]["Inter-slice interpolation information"]
                                prostate_interpolated_pts_np_arr = prostate_interslice_interpolation_information.interpolated_pts_np_arr
                                prostate_interpolated_pts_pcd = point_containment_tools.create_point_cloud(prostate_interpolated_pts_np_arr, color = np.array([0,1,1]))

                                # Shifted
                                shifted_bx_data_3darr_num_FANOVA_containment_sims_cutoff = shifted_bx_data_3darr[0:num_FANOVA_containment_simulations_input]
                                shifted_bx_pts_2d_arr_XYZ = np.reshape(shifted_bx_data_3darr_num_FANOVA_containment_sims_cutoff,(-1,3))

                                # Combine nominal and shifted
                                #combined_nominal_and_shifted_bx_pts_2d_arr_XYZ = np.vstack((unshifted_bx_sampled_pts_arr, shifted_bx_data_stacked_2darr_from_all_trials_3darray))
                                shifted_bx_pts_2d_arr_XY = shifted_bx_pts_2d_arr_XYZ[:,0:2]
                                shifted_bx_pts_2d_arr_Z = shifted_bx_pts_2d_arr_XYZ[:,2]
                        
                                shifted_nearest_interpolated_zslice_index_array, shifted_nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input(interpolated_zvlas_list,shifted_bx_pts_2d_arr_Z)
                        
                                shifted_bx_data_XY_interleaved_1darr = shifted_bx_pts_2d_arr_XY.flatten()
                                shifted_bx_data_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(shifted_bx_data_XY_interleaved_1darr)

                                specific_non_bx_struct_roi = specific_non_bx_structure["ROI"]
                                specific_non_bx_struct_refnum = specific_non_bx_structure["Ref #"]
                                structure_info = [specific_non_bx_struct_roi,
                                                non_bx_struct_type,
                                                specific_non_bx_struct_refnum,
                                                specific_non_bx_structure_index
                                                ]
                                
                                containment_info_grand_cudf_dataframe = point_containment_tools.cuspatial_points_contained_FANOVA(non_bx_struct_zslices_polygons_cuspatial_geoseries,
                                shifted_bx_data_XY_cuspatial_geoseries_points, 
                                shifted_bx_pts_2d_arr_XYZ, 
                                shifted_nearest_interpolated_zslice_index_array,
                                shifted_nearest_interpolated_zslice_vals_array,
                                non_bx_struct_max_zval,
                                non_bx_struct_min_zval, 
                                num_sample_pts_in_bx,
                                num_FANOVA_containment_simulations_input,
                                structure_info,
                                matrix_key
                                )

                                relative_structure_containment_results_data_frames_list.append(containment_info_grand_cudf_dataframe)

                                if (show_fanova_containment_demonstration_plots == True) & (non_bx_struct_type == 'DIL ref'):
                                    for trial_num, single_trial_shifted_bx_data_arr in enumerate(shifted_bx_data_3darr_num_FANOVA_containment_sims_cutoff):
                                        bx_test_pts_color_R = containment_info_grand_cudf_dataframe[containment_info_grand_cudf_dataframe["Trial num"] == trial_num+1]["Pt clr R"].to_numpy()
                                        bx_test_pts_color_G = containment_info_grand_cudf_dataframe[containment_info_grand_cudf_dataframe["Trial num"] == trial_num+1]["Pt clr G"].to_numpy()
                                        bx_test_pts_color_B = containment_info_grand_cudf_dataframe[containment_info_grand_cudf_dataframe["Trial num"] == trial_num+1]["Pt clr B"].to_numpy()
                                        bx_test_pts_color_arr = np.empty([num_sample_pts_in_bx,3])
                                        bx_test_pts_color_arr[:,0] = bx_test_pts_color_R
                                        bx_test_pts_color_arr[:,1] = bx_test_pts_color_G
                                        bx_test_pts_color_arr[:,2] = bx_test_pts_color_B
                                        structure_and_bx_shifted_bx_pcd = point_containment_tools.create_point_cloud_with_colors_array(single_trial_shifted_bx_data_arr, bx_test_pts_color_arr)
                                        plotting_funcs.plot_geometries(structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, label='Unknown')
                                        #plotting_funcs.plot_geometries(structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd, label='Unknown')
                                        #plotting_funcs.plot_two_views_side_by_side([structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd], containment_views_jsons_paths_list[0], [structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd], containment_views_jsons_paths_list[1])
                                        #plotting_funcs.plot_two_views_side_by_side([structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd], containment_views_jsons_paths_list[2], [structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd], containment_views_jsons_paths_list[3])
                        
                        structures_progress.update(testing_each_fanova_matrix_containment_task, advance=1)
                    
                    
                    # concatenate containment results into a single dataframe
                    containment_info_grand_all_structures_cudf_dataframe = cudf.concat(relative_structure_containment_results_data_frames_list, ignore_index=True)
                    del relative_structure_containment_results_data_frames_list
                    # Update the master dictionary
                    master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: sim containment raw results dataframe"] = containment_info_grand_all_structures_cudf_dataframe.to_pandas()

                    structures_progress.remove_task(testing_each_fanova_matrix_containment_task)
                    
                    biopsies_progress.update(testing_biopsy_containment_task, advance=1)
                biopsies_progress.remove_task(testing_biopsy_containment_task)

                patients_progress.update(testing_biopsy_containment_patient_task, advance = 1)
                completed_progress.update(testing_biopsy_containment_patient_task_completed, advance = 1)
            patients_progress.update(testing_biopsy_containment_patient_task, visible = False)
            completed_progress.update(testing_biopsy_containment_patient_task_completed, visible = True)
            live_display.refresh()


            #live_display.stop()
            if plot_cupy_fanova_containment_distribution_results == True:
                plotting_biopsy_containment_cuspatial_patient_task = patients_progress.add_task("[red]Plotting containment (cuspatial) results...", total=num_patients)
                plotting_biopsy_containment_cuspatial_patient_task_completed = completed_progress.add_task("[green]Plotting containment (cuspatial) results", total=num_patients, visible = False)
                for patientUID,pydicom_item in master_structure_reference_dict.items():
                    structure_organized_for_bx_data_blank_dict = MC_simulator_convex.create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
                    patients_progress.update(testing_biopsy_containment_patient_task, description = "[red]Testing biopsy containment (cuspatial) [{}]...".format(patientUID))
                    for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                        shifted_bx_data_dict = specific_bx_structure["FANOVA: bx only shifted 3darr dict"]
                        containment_info_grand_all_structures_cudf_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sim containment raw results dataframe"])
                        for relative_structure_info in structure_organized_for_bx_data_blank_dict.keys():
                            structure_roi = relative_structure_info[0]
                            non_bx_structure_type = relative_structure_info[1]
                            structure_index = relative_structure_info[3]

                            # Extract relative structure point cloud
                            non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                            non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                            non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))
                            
                            for matrix_key in shifted_bx_data_dict.keys():
                                grand_cudf_dataframe = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                                & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
                                                                                & (containment_info_grand_all_structures_cudf_dataframe["Matrix key"] == matrix_key)
                                                                                ].reset_index()

                                # Extract and rebuild pts color data
                                bx_test_pts_color_R = grand_cudf_dataframe["Pt clr R"].to_numpy()
                                bx_test_pts_color_G = grand_cudf_dataframe["Pt clr G"].to_numpy()
                                bx_test_pts_color_B = grand_cudf_dataframe["Pt clr B"].to_numpy()
                                bx_test_pts_color_arr = np.empty([len(grand_cudf_dataframe.index),3])
                                bx_test_pts_color_arr[:,0] = bx_test_pts_color_R
                                bx_test_pts_color_arr[:,1] = bx_test_pts_color_G
                                bx_test_pts_color_arr[:,2] = bx_test_pts_color_B

                                # Extract and rebuild pts vector data
                                bx_test_pts_X = grand_cudf_dataframe["Test pt X"].to_numpy()
                                bx_test_pts_Y = grand_cudf_dataframe["Test pt Y"].to_numpy()
                                bx_test_pts_Z = grand_cudf_dataframe["Test pt Z"].to_numpy()
                                bx_test_pts_arr = np.empty([len(grand_cudf_dataframe.index),3])
                                bx_test_pts_arr[:,0] = bx_test_pts_X
                                bx_test_pts_arr[:,1] = bx_test_pts_Y
                                bx_test_pts_arr[:,2] = bx_test_pts_Z

                                # Create point cloud
                                colored_bx_test_pts_pcd = point_containment_tools.create_point_cloud_with_colors_array(bx_test_pts_arr, bx_test_pts_color_arr)

                                patients_progress.stop_task(plotting_biopsy_containment_cuspatial_patient_task)
                                completed_progress.stop_task(plotting_biopsy_containment_cuspatial_patient_task_completed)
                                stopwatch.stop()
                                plotting_funcs.plot_geometries(colored_bx_test_pts_pcd, non_bx_struct_interpolated_pts_pcd, label='Unknown')
                                stopwatch.start()
                                patients_progress.start_task(plotting_biopsy_containment_cuspatial_patient_task)
                                completed_progress.start_task(plotting_biopsy_containment_cuspatial_patient_task_completed)
                    
                    patients_progress.update(plotting_biopsy_containment_cuspatial_patient_task, advance = 1)
                    completed_progress.update(plotting_biopsy_containment_cuspatial_patient_task_completed, advance = 1)
                patients_progress.update(plotting_biopsy_containment_cuspatial_patient_task, visible = False)
                completed_progress.update(plotting_biopsy_containment_cuspatial_patient_task_completed, visible = True)
                live_display.refresh()



            #live_display.stop()
            # Note that the matrix specific results 
            matrix_specific_results_dict_empty = {"Total successes list (by trial)": None, 
                                                    "Binomial estimator list (by trial)": None, 
                                                    "Confidence interval 95 (containment) list": None, 
                                                    "Standard error (containment) list": None,
                                                    "Global mean binomial estimator (all trials)": None
                                                    }
            compiling_results_patient_containment_task = patients_progress.add_task("[red]Compiling MC results ...", total=num_patients)
            compiling_results_patient_containment_task_completed = completed_progress.add_task("[green]Compiling MC results", total=num_patients, visible = False)  
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                patients_progress.update(compiling_results_patient_containment_task, description = "[red]Compiling MC results [{}]...".format(patientUID), total=num_patients)
                structure_organized_for_bx_data_blank_dict = MC_simulator_convex.create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)           
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                compiling_results_biopsy_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    biopsies_progress.update(compiling_results_biopsy_containment_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                    containment_info_grand_all_structures_cudf_dataframe = cudf.from_pandas(specific_bx_structure["FANOVA: sim containment raw results dataframe"])
                    shifted_bx_data_dict = specific_bx_structure["FANOVA: bx only shifted 3darr dict"]
                    fanova_compiled_results_for_fixed_bx_dict = copy.deepcopy(structure_organized_for_bx_data_blank_dict)
                    
                    sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                    sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                    sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs
                    compiling_results_each_non_bx_structure_containment_task = structures_progress.add_task("[cyan]~~For each structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                    for structure_info in fanova_compiled_results_for_fixed_bx_dict.keys():
                        structure_roi = structure_info[0]
                        non_bx_structure_type = structure_info[1]
                        structure_index = structure_info[3]

                        structures_progress.update(compiling_results_each_non_bx_structure_containment_task, description = "[cyan]~~For each structure [{}]...".format(structure_roi), total=sp_patient_total_num_non_BXs)

                        fanova_compiled_results_for_fixed_bx_dict[structure_info] = {matrix_key: None for matrix_key in shifted_bx_data_dict.keys()}
                        
                        for matrix_key in shifted_bx_data_dict.keys():
                            matrix_specific_results_dict = copy.deepcopy(matrix_specific_results_dict_empty)
                            # Shifted, note that we now want to count the total success within a single trial, and then divide by the number of sampled biopsy points to obtain the binomial estimator
                            bx_containment_counter_by_org_pt_ind_list = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                            & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
                                                                            & (containment_info_grand_all_structures_cudf_dataframe["Matrix key"] == matrix_key)
                                                                            ].reset_index()[
                                                                                ["Pt contained bool","Trial num"]
                                                                                ].groupby(["Trial num"]).sum().sort_index().to_numpy().T.flatten(order = 'C').tolist()
                            matrix_specific_results_dict["Total successes list (by trial)"] = bx_containment_counter_by_org_pt_ind_list                    
                            bx_containment_binomial_estimator_by_org_pt_ind_list = [x/num_sample_pts_in_bx for x in bx_containment_counter_by_org_pt_ind_list]
                            matrix_specific_results_dict["Binomial estimator list (by trial)"] = bx_containment_binomial_estimator_by_org_pt_ind_list
                            matrix_specific_results_dict["Global mean binomial estimator (all trials)"] = np.mean(bx_containment_binomial_estimator_by_org_pt_ind_list)

                            fanova_compiled_results_for_fixed_bx_dict[structure_info][matrix_key] = matrix_specific_results_dict

                        structures_progress.update(compiling_results_each_non_bx_structure_containment_task, advance=1)
                    
                    
                    del containment_info_grand_all_structures_cudf_dataframe
                    master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: sim containment raw results dataframe"] = 'Deleted'

                    structures_progress.remove_task(compiling_results_each_non_bx_structure_containment_task)
                    biopsies_progress.update(compiling_results_biopsy_containment_task, advance = 1) 
                    master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: compiled sim results"] = fanova_compiled_results_for_fixed_bx_dict
                biopsies_progress.remove_task(compiling_results_biopsy_containment_task) 
                patients_progress.update(compiling_results_patient_containment_task, advance = 1) 
                completed_progress.update(compiling_results_patient_containment_task_completed, advance = 1)
            patients_progress.update(compiling_results_patient_containment_task, visible = False) 
            completed_progress.update(compiling_results_patient_containment_task_completed, visible = True)
            live_display.refresh()

            
            
            #live_display.stop()
            calculating_sobol_containment_patient_task = patients_progress.add_task("[red]Calculating Sobol indices (containment) ...", total=num_patients)
            calculating_sobol_containment_patient_task_completed = completed_progress.add_task("[green]Calculating Sobol indices (containment)", total=num_patients, visible = False)  
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                patients_progress.update(calculating_sobol_containment_patient_task, description = "[red]Calculating Sobol indices (containment) [{}]...".format(patientUID), total=num_patients)
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                calculating_sobol_containment_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
                
                structure_organized_for_bx_data_blank_dict = MC_simulator_convex.create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)           

                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    biopsies_progress.update(calculating_sobol_containment_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                    
                    sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                    sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                    sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

                    fanova_sobol_indices_for_fixed_bx_dict = copy.deepcopy(structure_organized_for_bx_data_blank_dict)
                    
                    calculating_sobol_containment_each_non_bx_structure_task = structures_progress.add_task("[cyan]~~For each structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                    
                    fanova_compiled_results_for_fixed_bx_dict = specific_bx_structure["FANOVA: compiled sim results"]
                    for structure_info, fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict in fanova_compiled_results_for_fixed_bx_dict.items():
                        structure_roi = structure_info[0]
                        non_bx_structure_type = structure_info[1]
                        structure_index = structure_info[3]

                        structures_progress.update(calculating_sobol_containment_each_non_bx_structure_task, description = "[cyan]~~For each structure [{}]...".format(structure_roi), total=sp_patient_total_num_non_BXs)

                        ga = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['A']["Binomial estimator list (by trial)"]
                        gb = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['B']["Binomial estimator list (by trial)"]
                        
                        # for the scipy sobol indices function, the dictionary below is fed to it, and the arrays must have very specific shapes
                        fanova_containment_for_scipy_sobol_func_dict = {'f_A': np.array(ga).reshape((1,-1)),
                                                                        'f_B': np.array(gb).reshape((1,-1)),
                                                                        }
                        
                        fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_subset_dict = copy.deepcopy(fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict)
                        fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_subset_dict.pop('A')
                        fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_subset_dict.pop('B')
                        gab = np.empty((len(fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_subset_dict),1,num_FANOVA_containment_simulations_input))
                        
                        for matrix_key,fanova_compiles_results_sp_matrix in fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_subset_dict.items():     
                            gj = fanova_compiles_results_sp_matrix["Binomial estimator list (by trial)"]
                            gab[int(matrix_key)] = gj   
                        
                        fanova_containment_for_scipy_sobol_func_dict['f_AB'] = gab

                        sobol_indices = scipy.stats.sobol_indices(func = fanova_containment_for_scipy_sobol_func_dict, n=num_FANOVA_containment_simulations_input)
                        bootstrap_sobol_indices = sobol_indices.bootstrap(confidence_level = sobol_indices_bootstrap_conf_interval, 
                                                                        n_resamples = num_sobol_bootstraps
                                                                        )
                        
                        fanova_sobol_indices_for_fixed_bx_dict[structure_info] = {"Indices result": sobol_indices,
                                                                                "Bootstrap result": bootstrap_sobol_indices
                                                                                }

                        structures_progress.update(calculating_sobol_containment_each_non_bx_structure_task, advance=1)
                    
                    master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: sobol indices (containment)"] = fanova_sobol_indices_for_fixed_bx_dict

                    structures_progress.remove_task(calculating_sobol_containment_each_non_bx_structure_task)
                    biopsies_progress.update(calculating_sobol_containment_biopsy_task, advance = 1) 
                biopsies_progress.remove_task(calculating_sobol_containment_biopsy_task) 
                patients_progress.update(calculating_sobol_containment_patient_task, advance = 1) 
                completed_progress.update(calculating_sobol_containment_patient_task_completed, advance = 1)
            patients_progress.update(calculating_sobol_containment_patient_task, visible = False) 
            completed_progress.update(calculating_sobol_containment_patient_task_completed, visible = True)
            live_display.refresh()


            #live_display.stop()
            creating_sobol_containment_dataframes_patient_task = patients_progress.add_task("[red]Creating Sobol containment DFs ...", total=num_patients)
            creating_sobol_containment_dataframes_patient_task_completed = completed_progress.add_task("[green]Creating Sobol containment DFs", total=num_patients, visible = False)  
            num_variance_vars = master_structure_info_dict["Global"]["FANOVA: num variance vars"] 
            fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA: sobol var names by index"]
            for patientUID,pydicom_item in master_structure_reference_dict.items():  
                patients_progress.update(creating_sobol_containment_dataframes_patient_task, description = "[red]Creating Sobol containment DFs...".format(patientUID), total=num_patients)

                for specific_bx_structure in pydicom_item[bx_ref]:
                    bx_struct_roi = specific_bx_structure["ROI"]
                    simulated_bx_bool =  specific_bx_structure["Simulated bool"]
                    fanova_sobol_indices_for_fixed_bx_containment_dict = specific_bx_structure["FANOVA: sobol indices (containment)"]

                    num_rel_structures_in_in_sp_bx = len(fanova_sobol_indices_for_fixed_bx_containment_dict)
                    
                    cumulative_containment_sobol_indices_first_order_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    cumulative_containment_sobol_indices_first_order_se_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    cumulative_containment_sobol_indices_first_order_boot_low_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    cumulative_containment_sobol_indices_first_order_boot_high_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))

                    cumulative_containment_sobol_indices_total_order_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    cumulative_containment_sobol_indices_total_order_se_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    cumulative_containment_sobol_indices_total_order_boot_low_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    cumulative_containment_sobol_indices_total_order_boot_high_arr = np.empty((num_rel_structures_in_in_sp_bx,num_variance_vars))
                    
                    associated_relative_structure_by_row_index_list = [None]*num_rel_structures_in_in_sp_bx
                    associated_relative_structure_index_by_row_index_list = [None]*num_rel_structures_in_in_sp_bx
                    associated_relative_structure_type_by_row_index_list = [None]*num_rel_structures_in_in_sp_bx

                    
                    #specific_bx_structure["FANOVA: sobol indices (dose)"] = fanova_sobol_indices_for_fixed_bx_dict
                    for rel_struct_index, (structure_info, fanova_containment_indices_for_fixed_bx_for_fixed_relative_structure_dict) in enumerate(fanova_sobol_indices_for_fixed_bx_containment_dict.items()):
                        structure_roi = structure_info[0]
                        non_bx_structure_type = structure_info[1]
                        structure_index = structure_info[3]
                        
                        sp_indices = fanova_containment_indices_for_fixed_bx_for_fixed_relative_structure_dict["Indices result"]
                        sp_bootstrap = fanova_containment_indices_for_fixed_bx_for_fixed_relative_structure_dict["Bootstrap result"]
                        
                        cumulative_containment_sobol_indices_first_order_arr[rel_struct_index,:] = sp_indices.first_order
                        cumulative_containment_sobol_indices_first_order_se_arr[rel_struct_index,:] = sp_bootstrap.first_order.standard_error
                        cumulative_containment_sobol_indices_first_order_boot_low_arr[rel_struct_index,:] = sp_bootstrap.first_order.confidence_interval.low                   
                        cumulative_containment_sobol_indices_first_order_boot_high_arr[rel_struct_index,:] = sp_bootstrap.first_order.confidence_interval.high   
                        
                        cumulative_containment_sobol_indices_total_order_arr[rel_struct_index,:] = sp_indices.total_order
                        cumulative_containment_sobol_indices_total_order_se_arr[rel_struct_index,:] = sp_bootstrap.total_order.standard_error
                        cumulative_containment_sobol_indices_total_order_boot_low_arr[rel_struct_index,:] = sp_bootstrap.total_order.confidence_interval.low                   
                        cumulative_containment_sobol_indices_total_order_boot_high_arr[rel_struct_index,:] = sp_bootstrap.total_order.confidence_interval.high  
                        
                        associated_relative_structure_by_row_index_list[rel_struct_index] = structure_roi
                        associated_relative_structure_index_by_row_index_list[rel_struct_index] = structure_index
                        associated_relative_structure_type_by_row_index_list[rel_struct_index] = non_bx_structure_type

                    sp_bx_sobol_containment_for_dataframe_dict = {"Patient": patientUID,
                                                                "Bx ROI": bx_struct_roi,
                                                                "Simulated bx bool": simulated_bx_bool,
                                                                "Relative structure ROI": associated_relative_structure_by_row_index_list,
                                                                "Relative structure index": associated_relative_structure_index_by_row_index_list,
                                                                "Relative structure type": associated_relative_structure_type_by_row_index_list
                                                                }
                    
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' FO': cumulative_containment_sobol_indices_first_order_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' TO': cumulative_containment_sobol_indices_total_order_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' FO SE': cumulative_containment_sobol_indices_first_order_se_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' TO SE': cumulative_containment_sobol_indices_total_order_se_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' FO CI low': cumulative_containment_sobol_indices_first_order_boot_low_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' FO CI high': cumulative_containment_sobol_indices_first_order_boot_high_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' TO CI low': cumulative_containment_sobol_indices_total_order_boot_low_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_containment_for_dataframe_dict.update({name+' TO CI high': cumulative_containment_sobol_indices_total_order_boot_high_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})

                    sp_bx_sobol_containment_dataframe = cudf.DataFrame(sp_bx_sobol_containment_for_dataframe_dict)
                    sp_bx_sobol_containment_dataframe_no_na = sp_bx_sobol_containment_dataframe.fillna(0)

                    specific_bx_structure["FANOVA: sobol containment dataframe"] = sp_bx_sobol_containment_dataframe_no_na.to_pandas()
                    del sp_bx_sobol_containment_dataframe
                    del sp_bx_sobol_containment_dataframe_no_na

                patients_progress.update(creating_sobol_containment_dataframes_patient_task, advance = 1) 
                completed_progress.update(creating_sobol_containment_dataframes_patient_task_completed, advance = 1)
            patients_progress.update(creating_sobol_containment_dataframes_patient_task, visible = False) 
            completed_progress.update(creating_sobol_containment_dataframes_patient_task_completed, visible = True)
            live_display.refresh()


        """
        calculating_sobol_containment_patient_task = patients_progress.add_task("[red]Calculating Sobol indices (containment) ...", total=num_patients)
        calculating_sobol_containment_patient_task_completed = completed_progress.add_task("[green]Calculating Sobol indices (containment)", total=num_patients, visible = False)  
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calculating_sobol_containment_patient_task, description = "[red]Calculating Sobol indices (containment) [{}]...".format(patientUID), total=num_patients)
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            calculating_sobol_containment_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            
            structure_organized_for_bx_data_blank_dict = MC_simulator_convex.create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)           

            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(calculating_sobol_containment_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

                fanova_sobol_indices_for_fixed_bx_dict = copy.deepcopy(structure_organized_for_bx_data_blank_dict)
                
                calculating_sobol_containment_each_non_bx_structure_task = structures_progress.add_task("[cyan]~~For each structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                
                fanova_compiled_results_for_fixed_bx_dict = specific_bx_structure["FANOVA: compiled sim results"]
                for structure_info, fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict in fanova_compiled_results_for_fixed_bx_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_index = structure_info[3]

                    structures_progress.update(calculating_sobol_containment_each_non_bx_structure_task, description = "[cyan]~~For each structure [{}]...".format(structure_roi), total=sp_patient_total_num_non_BXs)

                    ga = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['A']["Binomial estimator list (by trial)"]
                    ga_mean = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['A']["Global mean binomial estimator (all trials)"]
                    gb = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['B']["Binomial estimator list (by trial)"]
                    gb_mean = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['B']["Global mean binomial estimator (all trials)"]
                    sobol_indices = {}
                    for matrix_key,fanova_compiles_results_sp_matrix in fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict.items():
                        if (matrix_key == 'A') or (matrix_key == 'B'):
                            continue
                        
                        gj = fanova_compiles_results_sp_matrix["Binomial estimator list (by trial)"]
                        sj_val = fanova_mathfuncs.sj(ga,gb,gj,ga_mean,gb_mean)

                        sobol_indices['S'+matrix_key] = sj_val

                        sj_tot_val = fanova_mathfuncs.sj_tot(ga,gb,gj, ga_mean, gb_mean)

                        sobol_indices['S_'+matrix_key+' tot'] = sj_tot_val

                    fanova_sobol_indices_for_fixed_bx_dict[structure_info] = sobol_indices

                    structures_progress.update(calculating_sobol_containment_each_non_bx_structure_task, advance=1)
                
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: sobol indices (containment)"] = fanova_sobol_indices_for_fixed_bx_dict

                structures_progress.remove_task(calculating_sobol_containment_each_non_bx_structure_task)
                biopsies_progress.update(calculating_sobol_containment_biopsy_task, advance = 1) 
            biopsies_progress.remove_task(calculating_sobol_containment_biopsy_task) 
            patients_progress.update(calculating_sobol_containment_patient_task, advance = 1) 
            completed_progress.update(calculating_sobol_containment_patient_task_completed, advance = 1)
        patients_progress.update(calculating_sobol_containment_patient_task, visible = False) 
        completed_progress.update(calculating_sobol_containment_patient_task_completed, visible = True)
        live_display.refresh()


        bootstrapping_sobol_containment_patient_task = patients_progress.add_task("[red]Bootstrapping Sobol indices (containment) ...", total=num_patients)
        bootstrapping_sobol_containment_patient_task_completed = completed_progress.add_task("[green]Bootstrapping Sobol indices (containment)", total=num_patients, visible = False)  
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(bootstrapping_sobol_containment_patient_task, description = "[red]Calculating Sobol indices (containment) [{}]...".format(patientUID), total=num_patients)
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            bootstrapping_sobol_containment_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            
            structure_organized_for_bx_data_blank_dict = MC_simulator_convex.create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)           

            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(bootstrapping_sobol_containment_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

                fanova_sobol_indices_for_fixed_bx_dict = copy.deepcopy(structure_organized_for_bx_data_blank_dict)
                
                bootstrapping_sobol_containment_each_non_bx_structure_task = structures_progress.add_task("[cyan]~~For each structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                
                fanova_compiled_results_for_fixed_bx_dict = specific_bx_structure["FANOVA: compiled sim results"]
                for structure_info, fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict in fanova_compiled_results_for_fixed_bx_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_index = structure_info[3]

                    structures_progress.update(bootstrapping_sobol_containment_each_non_bx_structure_task, description = "[cyan]~~For each structure [{}]...".format(structure_roi), total=sp_patient_total_num_non_BXs)
                    sobol_indices = {}
                    for bootstrap_index in range(0,num_sobol_bootstraps):
                        ga = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['A']["Binomial estimator list (by trial)"]
                        ga_resampled = np.random.choice(ga,len(ga),replace = True)
                        ga_resampled_mean = np.mean(ga_resampled)
                        gb = fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict['B']["Binomial estimator list (by trial)"]
                        gb_resampled = np.random.choice(gb,len(gb),replace = True)
                        gb_resampled_mean = np.mean(gb_resampled)
                        
                        for matrix_key,fanova_compiles_results_sp_matrix in fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict.items():
                            if (matrix_key == 'A') or (matrix_key == 'B'):
                                continue
                            sobol_indices['S'+matrix_key] = np.empty(num_sobol_bootstraps)
                            sobol_indices['S_'+matrix_key+' tot'] = np.empty(num_sobol_bootstraps)

                        for matrix_key,fanova_compiles_results_sp_matrix in fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict.items():
                            if (matrix_key == 'A') or (matrix_key == 'B'):
                                continue
                            
                            gj = fanova_compiles_results_sp_matrix["Binomial estimator list (by trial)"]
                            gj_resampled = np.random.choice(gj,len(gj),replace = True)

                            sj_val = fanova_mathfuncs.sj(ga_resampled,gb_resampled,gj_resampled,ga_resampled_mean,gb_resampled_mean)

                            sobol_indices['S'+matrix_key][bootstrap_index] = sj_val

                            sj_tot_val = fanova_mathfuncs.sj_tot(ga_resampled,gb_resampled,gj_resampled, ga_resampled_mean, gb_resampled_mean)

                            sobol_indices['S_'+matrix_key+' tot'][bootstrap_index] = sj_tot_val

                    for matrix_key,fanova_compiles_results_sp_matrix in fanova_compiled_results_for_fixed_bx_for_fixed_relative_structure_dict.items():
                        if (matrix_key == 'A') or (matrix_key == 'B'):
                            continue
                        # calculate the mean of the sobol indices
                        sobol_indices['S'+matrix_key+' mean'] = np.mean(sobol_indices['S'+matrix_key])
                        sobol_indices['S'+matrix_key+' tot mean'] = np.mean(sobol_indices['S_'+matrix_key+' tot'])
                        # standard error in the mean
                        sobol_indices['S'+matrix_key+' SE'] = np.std(sobol_indices['S'+matrix_key])/np.sqrt(sobol_indices['S'+matrix_key].shape[0])   
                        sobol_indices['S'+matrix_key+' tot SE'] = np.std(sobol_indices['S_'+matrix_key+' tot'])/np.sqrt(sobol_indices['S_'+matrix_key+' tot'].shape[0])   
                    
                    fanova_sobol_indices_for_fixed_bx_dict[structure_info] = sobol_indices

                    structures_progress.update(bootstrapping_sobol_containment_each_non_bx_structure_task, advance=1)
                
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: sobol indices bootstrapped (containment)"] = fanova_sobol_indices_for_fixed_bx_dict

                structures_progress.remove_task(bootstrapping_sobol_containment_each_non_bx_structure_task)
                biopsies_progress.update(bootstrapping_sobol_containment_biopsy_task, advance = 1) 
            biopsies_progress.remove_task(bootstrapping_sobol_containment_biopsy_task) 
            patients_progress.update(bootstrapping_sobol_containment_patient_task, advance = 1) 
            completed_progress.update(bootstrapping_sobol_containment_patient_task_completed, advance = 1)
        patients_progress.update(bootstrapping_sobol_containment_patient_task, visible = False) 
        completed_progress.update(bootstrapping_sobol_containment_patient_task_completed, visible = True)
        live_display.refresh()
        """


        if perform_dose_fanova == True:
            #live_display.stop()
            calc_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]FANOVA NN dosimetric localization [{}]...".format("initializing"), total=num_patients)
            calc_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]FANOVA NN dosimetric localization", total=num_patients, visible = False)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                patients_progress.update(calc_dose_NN_biopsy_containment_task, description = "[red]FANOVA NN dosimetric localization [{}]...".format(patientUID))
                
                # create KDtree for dose data
                creating_kd_tree_task = indeterminate_progress_sub.add_task("[cyan]~~Creating KDTree [{}]...".format(patientUID), total = None)
                
                dose_ref_dict = pydicom_item[dose_ref]
                phys_space_dose_map_3d_arr = dose_ref_dict["Dose phys space and pixel 3d arr"]
                phys_space_dose_map_3d_arr_flattened = np.reshape(phys_space_dose_map_3d_arr, (-1,7) , order = 'C') # turn the data into a 2d array
                phys_space_dose_map_phys_coords_2d_arr = phys_space_dose_map_3d_arr_flattened[:,3:6] 
                phys_space_dose_map_dose_2d_arr = phys_space_dose_map_3d_arr_flattened[:,6] 
                dose_data_KDtree = scipy.spatial.KDTree(phys_space_dose_map_phys_coords_2d_arr)
                dose_ref_dict["KDtree"] = dose_data_KDtree

                indeterminate_progress_sub.remove_task(creating_kd_tree_task)

                

                

                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                dosimetric_calc_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)           
                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    biopsies_progress.update(dosimetric_calc_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                    
                    
                    #bx_only_shifted_3darr = cp.asnumpy(specific_bx_structure["MC data: bx only shifted 3darr"]) # note that the 3rd dimension slices are each MC trial
                    #bx_only_shifted_3darr_cutoff = bx_only_shifted_3darr[0:num_FANOVA_dose_simulations_input]
                    unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                    #unshifted_bx_sampled_pts_arr_3darr = np.expand_dims(unshifted_bx_sampled_pts_arr, axis=0)
                    #nominal_and_bx_only_shifted_3darr = np.concatenate((unshifted_bx_sampled_pts_arr_3darr,bx_only_shifted_3darr_cutoff))

                    shifted_bx_data_dict = specific_bx_structure["FANOVA: bx only shifted 3darr dict"]
                    
                    ###
                    testing_each_fanova_matrix_dose_task = structures_progress.add_task("[cyan]~~For each FANOVA matrix [{}]...".format("initializing"), total=len(shifted_bx_data_dict))
                    
                    #fanova_NN_search_objects_by_matrix_dict = {}
                    
                    dose_vals_by_fanova_matrix_dict = {}
                    #nn_objs_by_fanova_matrix_dict = {}
                    for matrix_key, shifted_bx_data_3darr_cp in shifted_bx_data_dict.items():
                        structures_progress.update(testing_each_fanova_matrix_dose_task, description = "[cyan]~~For each FANOVA matrix [{}]...".format(matrix_key))

                        shifted_bx_data_3darr = cp.asnumpy(shifted_bx_data_3darr_cp)
                        del shifted_bx_data_3darr_cp  
                        shifted_bx_data_3darr_num_FANOVA_containment_sims_cutoff = shifted_bx_data_3darr[0:num_FANOVA_dose_simulations_input]

                        dosimetric_calc_parallel_task = indeterminate_progress_sub.add_task("[cyan]~~Conducting NN search [{}]...".format(specific_bx_structure_roi), total = None)
                        # non-parallel
                        dosimetric_localization_sp_matrix_all_trials_list = MC_simulator_convex.dosimetric_localization_non_parallel(shifted_bx_data_3darr_num_FANOVA_containment_sims_cutoff, 
                                                                                                                        specific_bx_structure, 
                                                                                                                        dose_ref_dict, 
                                                                                                                        dose_ref, 
                                                                                                                        phys_space_dose_map_phys_coords_2d_arr, 
                                                                                                                        phys_space_dose_map_dose_2d_arr, 
                                                                                                                        num_dose_calc_NN
                                                                                                                        )
                        
                        dosimetric_localization_all_MC_trials_list_NN_lists_only = [NN_parent_obj.NN_data_list for NN_parent_obj in dosimetric_localization_sp_matrix_all_trials_list]
                        dosimetric_localization_NN_child_objs_by_bx_point_sp_matrix_all_trials_list = list(zip(*dosimetric_localization_all_MC_trials_list_NN_lists_only))
                        dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_list = [[NN_child_obj.nearest_dose for NN_child_obj in fixed_bx_pt_NN_objs_list] for fixed_bx_pt_NN_objs_list in dosimetric_localization_NN_child_objs_by_bx_point_sp_matrix_all_trials_list]
                        dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr = np.array(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_list)
                
                        #nn_objs_by_fanova_matrix_dict[matrix_key] = dosimetric_localization_NN_child_objs_by_bx_point_sp_matrix_all_trials_list
                        dose_vals_by_fanova_matrix_dict[matrix_key] = dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr



                        if show_NN_FANOVA_dose_demonstration_plots == True:
                            # code for the plotting of the below NN search of sampled bx pts
                            lattice_dose_pcd = dose_ref_dict["Dose grid point cloud"]
                            thresholded_lattice_dose_pcd = dose_ref_dict["Dose grid point cloud thresholded"]
                            unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                            unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                            
                            # plot everything to make sure its working properly!
                            for mc_trial_NN_parent_dose_obj in dosimetric_localization_sp_matrix_all_trials_list:
                                NN_data_list_for_one_MC_trial = mc_trial_NN_parent_dose_obj.NN_data_list
                                NN_pts_on_comparison_struct_for_all_points_concatenated = np.empty([num_dose_calc_NN*num_sample_pts_per_bx,3])
                                NN_doses_on_comparison_struct_for_all_points_concatenated = np.empty([num_dose_calc_NN*num_sample_pts_per_bx])
                                queried_bx_pts_arr_concatenated = np.empty([num_sample_pts_per_bx,3])
                                queried_bx_pts_assigned_doses_arr_concatenated = np.empty([num_sample_pts_per_bx])
                                for sampled_pt_index,NN_child_dose_obj in enumerate(NN_data_list_for_one_MC_trial):
                                    NN_pts_array_on_dose_lattice = NN_child_dose_obj.NN_pt_on_comparison_struct
                                    NN_dose_array_on_dose_lattice = NN_child_dose_obj.NN_dose_on_comparison_struct
                                    queried_bx_pt_instance = NN_child_dose_obj.queried_BX_pt
                                    queried_bx_pt_assigned_dose_instance = NN_child_dose_obj.nearest_dose
                                    NN_pts_on_comparison_struct_for_all_points_concatenated[sampled_pt_index*num_dose_calc_NN:sampled_pt_index*num_dose_calc_NN+num_dose_calc_NN,:] = NN_pts_array_on_dose_lattice
                                    NN_doses_on_comparison_struct_for_all_points_concatenated[sampled_pt_index*num_dose_calc_NN:sampled_pt_index*num_dose_calc_NN+num_dose_calc_NN] = NN_dose_array_on_dose_lattice
                                    queried_bx_pts_arr_concatenated[sampled_pt_index] = queried_bx_pt_instance
                                    queried_bx_pts_assigned_doses_arr_concatenated[sampled_pt_index] = queried_bx_pt_assigned_dose_instance
                                
                                patients_progress.stop_task(calc_dose_NN_biopsy_containment_task)
                                completed_progress.stop_task(calc_dose_NN_biopsy_containment_task_complete)
                                stopwatch.stop()
                                #plotting_funcs.dose_point_cloud_with_dose_labels_for_animation(NN_pts_on_comparison_struct_for_all_points_concatenated, NN_doses_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, queried_bx_pts_assigned_doses_arr_concatenated, num_dose_calc_NN, draw_lines = True)
                                geometry_list_full_dose_lattice = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, lattice_dose_pcd, NN_pts_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, num_dose_calc_NN, draw_lines = True)
                                geometry_list_thresholded_dose_lattice = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, thresholded_lattice_dose_pcd, NN_pts_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, num_dose_calc_NN, draw_lines = True)
                                plotting_funcs.plot_two_views_side_by_side(geometry_list_thresholded_dose_lattice, dose_views_jsons_paths_list[0], geometry_list_thresholded_dose_lattice, dose_views_jsons_paths_list[1])
                                plotting_funcs.plot_two_views_side_by_side(geometry_list_thresholded_dose_lattice, dose_views_jsons_paths_list[2], geometry_list_thresholded_dose_lattice, dose_views_jsons_paths_list[3])
                                plotting_funcs.dose_point_cloud_with_dose_labels_for_animation_plotly(NN_pts_on_comparison_struct_for_all_points_concatenated, NN_doses_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, queried_bx_pts_assigned_doses_arr_concatenated, num_dose_calc_NN, aspect_mode_input = 'data', draw_lines = False, axes_visible=True)
                                plotting_funcs.dose_point_cloud_with_dose_labels_for_animation_plotly(NN_pts_on_comparison_struct_for_all_points_concatenated, NN_doses_on_comparison_struct_for_all_points_concatenated, queried_bx_pts_arr_concatenated, queried_bx_pts_assigned_doses_arr_concatenated, num_dose_calc_NN, aspect_mode_input = 'data', draw_lines = True, axes_visible=True)
                                stopwatch.start()
                                patients_progress.start_task(calc_dose_NN_biopsy_containment_task)
                                completed_progress.start_task(calc_dose_NN_biopsy_containment_task_complete)
                        else:
                            pass



                        
                        # parallel
                        """
                        dosimetric_localization_all_MC_trials_list = MC_simulator_convex.dosimetric_localization_parallel(parallel_pool, 
                                                                                                                        shifted_bx_data_3darr_num_FANOVA_containment_sims_cutoff, 
                                                                                                                        specific_bx_structure, 
                                                                                                                        dose_ref_dict, 
                                                                                                                        dose_ref, 
                                                                                                                        phys_space_dose_map_phys_coords_2d_arr, 
                                                                                                                        phys_space_dose_map_dose_2d_arr, 
                                                                                                                        num_dose_calc_NN
                                                                                                                        )
                        """
                        indeterminate_progress_sub.remove_task(dosimetric_calc_parallel_task)
                        
                        #fanova_NN_search_objects_by_matrix_dict[matrix_key] = dosimetric_localization_all_MC_trials_list
                        structures_progress.update(testing_each_fanova_matrix_dose_task, advance=1)
                    
                    
                    # Update master dictionary
                    #specific_bx_structure["FANOVA: Dose NN child obj for each sampled bx pt dict of lists (all trials)"] = nn_objs_by_fanova_matrix_dict
                    specific_bx_structure["FANOVA: Dose vals for each sampled bx pt dict of arrays (all trials)"] = dose_vals_by_fanova_matrix_dict
                    
                    
                    #specific_bx_structure['FANOVA: bx to dose NN search objects dict of lists'] = fanova_NN_search_objects_by_matrix_dict 

                    structures_progress.remove_task(testing_each_fanova_matrix_dose_task)
                    biopsies_progress.update(dosimetric_calc_biopsy_task, advance=1)
                biopsies_progress.remove_task(dosimetric_calc_biopsy_task)
                patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
                completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
            patients_progress.update(calc_dose_NN_biopsy_containment_task, visible = False)
            completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, visible = True)
            live_display.refresh()



            """
            compile_results_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]Compiling dosimetric localization results [{}]...".format("initializing"), total=num_patients)
            compile_results_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]Compiling dosimetric localization results", total=num_patients, visible = False)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                patients_progress.update(compile_results_dose_NN_biopsy_containment_task, description = "[red]Compiling dosimetric localization results [{}]...".format(patientUID))
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                compile_results_dose_NN_biopsy_containment_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)
                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                    fanova_NN_search_objects_by_matrix_dict = specific_bx_structure['FANOVA: bx to dose NN search objects dict of lists'] 
                    
                    testing_each_fanova_matrix_dose_task = structures_progress.add_task("[cyan]~~For each FANOVA matrix [{}]...".format("initializing"), total=len(shifted_bx_data_dict))

                    dose_vals_by_fanova_matrix_dict = {}
                    nn_objs_by_fanova_matrix_dict = {}

                    for matrix_key, dosimetric_localization_sp_matrix_all_trials_list in fanova_NN_search_objects_by_matrix_dict.items():
                        structures_progress.update(testing_each_fanova_matrix_dose_task, description = "[cyan]~~For each FANOVA matrix [{}]...".format(matrix_key))
                    
                        dosimetric_localization_all_MC_trials_list_NN_lists_only = [NN_parent_obj.NN_data_list for NN_parent_obj in dosimetric_localization_sp_matrix_all_trials_list]
                        dosimetric_localization_NN_child_objs_by_bx_point_sp_matrix_all_trials_list = list(zip(*dosimetric_localization_all_MC_trials_list_NN_lists_only))
                        dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_list = [[NN_child_obj.nearest_dose for NN_child_obj in fixed_bx_pt_NN_objs_list] for fixed_bx_pt_NN_objs_list in dosimetric_localization_NN_child_objs_by_bx_point_sp_matrix_all_trials_list]
                        dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr = np.array(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_list)
                
                        nn_objs_by_fanova_matrix_dict[matrix_key] = dosimetric_localization_NN_child_objs_by_bx_point_sp_matrix_all_trials_list
                        dose_vals_by_fanova_matrix_dict[matrix_key] = dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr
                        structures_progress.update(testing_each_fanova_matrix_dose_task, advance=1)


                    # Update master dictionary
                    specific_bx_structure["FANOVA: Dose NN child obj for each sampled bx pt dict of lists (all trials)"] = nn_objs_by_fanova_matrix_dict
                    specific_bx_structure["FANOVA: Dose vals for each sampled bx pt dict of arrays (all trials)"] = dose_vals_by_fanova_matrix_dict
                    
                    
                    structures_progress.remove_task(testing_each_fanova_matrix_dose_task)
                    biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, advance = 1)
                biopsies_progress.remove_task(compile_results_dose_NN_biopsy_containment_by_biopsy_task)    
                patients_progress.update(compile_results_dose_NN_biopsy_containment_task, advance=1)
                completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, advance=1)
            patients_progress.update(compile_results_dose_NN_biopsy_containment_task, visible = False)
            completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, visible = True)
            live_display.refresh()
            """

                            
            compile_results_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]Computing dosimetric FANOVA statistics [{}]...".format("initializing"), total=num_patients)
            compile_results_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]Computing dosimetric FANOVA statistics ", total=num_patients, visible = False)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                patients_progress.update(compile_results_dose_NN_biopsy_containment_task, description = "[red]Computing dosimetric FANOVA statistics  [{}]...".format(patientUID))
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                compile_results_dose_NN_biopsy_containment_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)
                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                    #fanova_NN_search_objects_by_matrix_dict = specific_bx_structure['FANOVA: bx to dose NN search objects dict of lists'] 
                    
                    testing_each_fanova_matrix_dose_task = structures_progress.add_task("[cyan]~~For each FANOVA matrix [{}]...".format("initializing"), total=len(shifted_bx_data_dict))

                    dose_vals_by_fanova_matrix_dict = specific_bx_structure["FANOVA: Dose vals for each sampled bx pt dict of arrays (all trials)"]
                    randomly_sampled_bx_pts_bx_coord_sys_arr = specific_bx_structure['Random uniformly sampled volume pts bx coord sys arr']
                    
                    ouput_function_dose_by_fanova_matrix = {}
                    for matrix_key, dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr in dose_vals_by_fanova_matrix_dict.items():
                        # for the dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr, the columns are trials, rows are a fixed bx point           

                        ouput_function_by_fanova_trial_sp_matrix_dict = {}
                        structures_progress.update(testing_each_fanova_matrix_dose_task, description = "[cyan]~~For each FANOVA matrix [{}]...".format(matrix_key))

                        
                        max_dose_val_of_each_trial_arr = np.max(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr, axis = 0)
                        min_dose_val_of_each_trial_arr = np.min(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr, axis = 0)
                        
                        max_min_difference_of_each_trial_arr = max_dose_val_of_each_trial_arr - min_dose_val_of_each_trial_arr
                        
                        bx_pt_index_of_max_dose_val_of_each_trial_arr = np.argmax(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr, axis = 0)
                        bx_pt_zval_of_max_dose_val_of_each_trial_arr = randomly_sampled_bx_pts_bx_coord_sys_arr[bx_pt_index_of_max_dose_val_of_each_trial_arr, 2]
                        
                        bx_pt_index_of_min_dose_val_of_each_trial_arr = np.argmin(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr, axis = 0)
                        bx_pt_zval_of_min_dose_val_of_each_trial_arr = randomly_sampled_bx_pts_bx_coord_sys_arr[bx_pt_index_of_min_dose_val_of_each_trial_arr, 2]

                        mean_dose_of_each_trial_arr = np.mean(dosimetric_localization_dose_vals_by_bx_point_sp_matrix_all_trials_arr, axis = 0)

                        ouput_function_by_fanova_trial_sp_matrix_dict["Mean dose arr"] = mean_dose_of_each_trial_arr
                        ouput_function_by_fanova_trial_sp_matrix_dict["Z-val of max dose arr"] = bx_pt_zval_of_max_dose_val_of_each_trial_arr
                        ouput_function_by_fanova_trial_sp_matrix_dict["Z-val of min dose arr"] = bx_pt_zval_of_min_dose_val_of_each_trial_arr
                        ouput_function_by_fanova_trial_sp_matrix_dict["Max-min dose arr"] = max_min_difference_of_each_trial_arr
                        ouput_function_by_fanova_trial_sp_matrix_dict["Max dose arr"] = max_dose_val_of_each_trial_arr
                        ouput_function_by_fanova_trial_sp_matrix_dict["Min dose arr"] = min_dose_val_of_each_trial_arr

                        ouput_function_dose_by_fanova_matrix[matrix_key] = ouput_function_by_fanova_trial_sp_matrix_dict

                        structures_progress.update(testing_each_fanova_matrix_dose_task, advance=1)


                    # Update master dictionary
                    specific_bx_structure["FANOVA: Dose function outputs dict"] = ouput_function_dose_by_fanova_matrix
                    
                    structures_progress.remove_task(testing_each_fanova_matrix_dose_task)
                    biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, advance = 1)
                biopsies_progress.remove_task(compile_results_dose_NN_biopsy_containment_by_biopsy_task)    
                patients_progress.update(compile_results_dose_NN_biopsy_containment_task, advance=1)
                completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, advance=1)
            patients_progress.update(compile_results_dose_NN_biopsy_containment_task, visible = False)
            completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, visible = True)
            live_display.refresh()

            #live_display.stop()
            calculating_sobol_dose_patient_task = patients_progress.add_task("[red]Calculating Sobol indices (dosimetry) ...", total=num_patients)
            calculating_sobol_dose_patient_task_completed = completed_progress.add_task("[green]Calculating Sobol indices (dosimetry)", total=num_patients, visible = False)  
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                patients_progress.update(calculating_sobol_dose_patient_task, description = "[red]Calculating Sobol indices (dosimetry) [{}]...".format(patientUID), total=num_patients)
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                calculating_sobol_dose_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
                

                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                    specific_bx_structure_roi = specific_bx_structure["ROI"]
                    biopsies_progress.update(calculating_sobol_dose_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                    
                    sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
                    sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                    sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs
                    
                    calculating_sobol_dose_each_non_bx_structure_task = structures_progress.add_task("[cyan]~~For each structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                    
                    ouput_function_dosimetry_by_fanova_matrix = specific_bx_structure["FANOVA: Dose function outputs dict"]
                    
                    
                
                    ga_dict = ouput_function_dosimetry_by_fanova_matrix['A']
                    ga_arr = np.empty((len(ga_dict),num_FANOVA_dose_simulations_input))
                    ouput_keys_by_matrix_dict = {}
                    ouput_keys_by_index_list = [None]*len(ga_dict)
                    for index,(output_func_key,output_func_arr) in enumerate(ga_dict.items()):
                        ouput_keys_by_index_list[index] = output_func_key
                        ga_arr[index,:] = output_func_arr
                    ouput_keys_by_matrix_dict['A'] = ouput_keys_by_index_list
                    
                    gb_dict = ouput_function_dosimetry_by_fanova_matrix['B']
                    ouput_keys_by_index_list = [None]*len(gb_dict)
                    gb_arr = np.empty((len(gb_dict),num_FANOVA_dose_simulations_input))
                    for index,(output_func_key,output_func_arr) in enumerate(gb_dict.items()):
                        ouput_keys_by_index_list[index] = output_func_key
                        gb_arr[index,:] = output_func_arr
                    ouput_keys_by_matrix_dict['B'] = ouput_keys_by_index_list
    
                    # for the scipy sobol indices function, the dictionary below is fed to it, and the arrays must have very specific shapes
                    fanova_dosimetry_for_scipy_sobol_func_dict = {'f_A': ga_arr,
                                                                    'f_B': gb_arr,
                                                                    }
                        
                    ouput_function_dosimetry_by_fanova_matrix_subset = copy.deepcopy(ouput_function_dosimetry_by_fanova_matrix)
                    ouput_function_dosimetry_by_fanova_matrix_subset.pop('A')
                    ouput_function_dosimetry_by_fanova_matrix_subset.pop('B')
                    gab_arr = np.empty((len(ouput_function_dosimetry_by_fanova_matrix_subset),len(gb_dict),num_FANOVA_dose_simulations_input))
                        
                    for matrix_key, fanova_compiled_dose_results_sp_matrix in ouput_function_dosimetry_by_fanova_matrix_subset.items():     
                        gab_constant_matrix_output_slice = np.empty((len(fanova_compiled_dose_results_sp_matrix),num_FANOVA_dose_simulations_input))
                        ouput_keys_by_index_list = [None]*len(fanova_compiled_dose_results_sp_matrix)
                        for index,(output_func_key,output_func_arr) in enumerate(fanova_compiled_dose_results_sp_matrix.items()):
                            ouput_keys_by_index_list[index] = output_func_key
                            gab_constant_matrix_output_slice[index,:] = output_func_arr
                        
                        ouput_keys_by_matrix_dict[matrix_key] = ouput_keys_by_index_list
                        gab_arr[int(matrix_key)] = gab_constant_matrix_output_slice  
                        
                    fanova_dosimetry_for_scipy_sobol_func_dict['f_AB'] = gab_arr

                    sobol_indices = scipy.stats.sobol_indices(func = fanova_dosimetry_for_scipy_sobol_func_dict, n=num_FANOVA_dose_simulations_input)
                    bootstrap_sobol_indices = sobol_indices.bootstrap(confidence_level = sobol_indices_bootstrap_conf_interval, 
                                                                        n_resamples = num_sobol_bootstraps
                                                                        )
                        
                    fanova_sobol_indices_for_fixed_bx_dict = {"Indices result": sobol_indices,
                                                                "Bootstrap result": bootstrap_sobol_indices,
                                                                "Output results key index dict": ouput_keys_by_matrix_dict
                                                                }

                    structures_progress.update(calculating_sobol_dose_each_non_bx_structure_task, advance=1)
                    
                    master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["FANOVA: sobol indices (dose)"] = fanova_sobol_indices_for_fixed_bx_dict

                    structures_progress.remove_task(calculating_sobol_dose_each_non_bx_structure_task)
                    biopsies_progress.update(calculating_sobol_dose_biopsy_task, advance = 1) 
                biopsies_progress.remove_task(calculating_sobol_dose_biopsy_task) 
                patients_progress.update(calculating_sobol_dose_patient_task, advance = 1) 
                completed_progress.update(calculating_sobol_dose_patient_task_completed, advance = 1)
            patients_progress.update(calculating_sobol_dose_patient_task, visible = False) 
            completed_progress.update(calculating_sobol_dose_patient_task_completed, visible = True)
            live_display.refresh()



            creating_sobol_dosimetry_dataframes_patient_task = patients_progress.add_task("[red]Creating Sobol dosimetry DFs ...", total=num_patients)
            creating_sobol_dosimetry_dataframes_patient_task_completed = completed_progress.add_task("[green]Creating Sobol dosimetry DFs", total=num_patients, visible = False)  
            num_variance_vars = master_structure_info_dict["Global"]["FANOVA: num variance vars"] 
            fanova_sobol_indices_names_by_index = master_structure_info_dict["Global"]["FANOVA: sobol var names by index"]
            for patientUID,pydicom_item in master_structure_reference_dict.items():  
                patients_progress.update(creating_sobol_dosimetry_dataframes_patient_task, description = "[red]Creating Sobol dosimetry DFs...".format(patientUID), total=num_patients)

                for specific_bx_structure in pydicom_item[bx_ref]:
                    bx_struct_roi = specific_bx_structure["ROI"]
                    simulated_bx_bool =  specific_bx_structure["Simulated bool"]
                    fanova_sobol_indices_for_fixed_bx_dose_dict = specific_bx_structure["FANOVA: sobol indices (dose)"]

                    dose_indices_result = fanova_sobol_indices_for_fixed_bx_dose_dict["Indices result"]
                    dose_bootstrap_result = fanova_sobol_indices_for_fixed_bx_dose_dict["Bootstrap result"]
                    output_result_by_matrix_key_dict = fanova_sobol_indices_for_fixed_bx_dose_dict["Output results key index dict"] # note each key value should be identical
                    output_result_key_list = list(output_result_by_matrix_key_dict.values())[0] # just pick the first one since theyre all equal 

                    cumulative_dose_sobol_indices_first_order_arr = dose_indices_result.first_order
                    cumulative_dose_sobol_indices_first_order_se_arr = dose_bootstrap_result.first_order.standard_error
                    cumulative_dose_sobol_indices_first_order_boot_low_arr = dose_bootstrap_result.first_order.confidence_interval.low
                    cumulative_dose_sobol_indices_first_order_boot_high_arr = dose_bootstrap_result.first_order.confidence_interval.high

                    cumulative_dose_sobol_indices_total_order_arr = dose_indices_result.total_order
                    cumulative_dose_sobol_indices_total_order_se_arr = dose_bootstrap_result.total_order.standard_error
                    cumulative_dose_sobol_indices_total_order_boot_low_arr = dose_bootstrap_result.total_order.confidence_interval.low
                    cumulative_dose_sobol_indices_total_order_boot_high_arr = dose_bootstrap_result.total_order.confidence_interval.high
                    
                    """
                    associated_relative_structure_by_row_index_list = [None]*num_rel_structures_in_in_sp_bx
                    associated_relative_structure_index_by_row_index_list = [None]*num_rel_structures_in_in_sp_bx
                    associated_relative_structure_type_by_row_index_list = [None]*num_rel_structures_in_in_sp_bx
                        
                    associated_relative_structure_by_row_index_list[rel_struct_index] = structure_roi
                    associated_relative_structure_index_by_row_index_list[rel_struct_index] = structure_index
                    associated_relative_structure_type_by_row_index_list[rel_struct_index] = non_bx_structure_type
                    """

                    sp_bx_sobol_dose_for_dataframe_dict = {"Patient": patientUID,
                                                                "Bx ROI": bx_struct_roi,
                                                                "Simulated bx bool": simulated_bx_bool,
                                                                "Function output key": output_result_key_list
                                                                }
                    
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' FO': cumulative_dose_sobol_indices_first_order_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' TO': cumulative_dose_sobol_indices_total_order_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' FO SE': cumulative_dose_sobol_indices_first_order_se_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' TO SE': cumulative_dose_sobol_indices_total_order_se_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' FO CI low': cumulative_dose_sobol_indices_first_order_boot_low_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' FO CI high': cumulative_dose_sobol_indices_first_order_boot_high_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' TO CI low': cumulative_dose_sobol_indices_total_order_boot_low_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})
                    sp_bx_sobol_dose_for_dataframe_dict.update({name+' TO CI high': cumulative_dose_sobol_indices_total_order_boot_high_arr[:,index] for index,name in enumerate(fanova_sobol_indices_names_by_index)})

                    sp_bx_sobol_dose_dataframe = cudf.DataFrame(sp_bx_sobol_dose_for_dataframe_dict)
                    sp_bx_sobol_dose_dataframe_no_na = sp_bx_sobol_dose_dataframe.fillna(0)

                    specific_bx_structure["FANOVA: sobol dose dataframe"] = sp_bx_sobol_dose_dataframe_no_na.to_pandas()
                    del sp_bx_sobol_dose_dataframe_no_na
                    del sp_bx_sobol_dose_dataframe

                patients_progress.update(creating_sobol_dosimetry_dataframes_patient_task, advance = 1) 
                completed_progress.update(creating_sobol_dosimetry_dataframes_patient_task_completed, advance = 1)
            patients_progress.update(creating_sobol_dosimetry_dataframes_patient_task, visible = False) 
            completed_progress.update(creating_sobol_dosimetry_dataframes_patient_task_completed, visible = True)
            live_display.refresh()


        master_structure_info_dict['Global']['FANOVA sim performed'] = True
        master_structure_info_dict['Global']['FANOVA containment sim performed'] = perform_containment_fanova
        master_structure_info_dict['Global']['FANOVA dose sim performed'] = perform_dose_fanova
            