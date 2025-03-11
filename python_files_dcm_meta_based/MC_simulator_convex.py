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
import cuspatial
from shapely.geometry import Point, Polygon
import geopandas
import cudf
import cupy_functions
import cupy as cp
import itertools
import pandas
import misc_tools
import dosimetric_localizer
import dataframe_builders
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import polygon_dilation_helpers
import polygon_dilation_helpers_numpy
import polygon_dilation_helpers_cupy
import cProfile
import pstats
import io
from line_profiler import LineProfiler


def simulator_parallel(parallel_pool, 
                       live_display,
                       stopwatch, 
                       layout_groups, 
                       master_structure_reference_dict, 
                       structs_referenced_list,
                       structs_referenced_dict,
                       bx_ref,
                       oar_ref,
                       dil_ref,
                       rectum_ref,
                       urethra_ref, 
                       dose_ref,
                       plan_ref, 
                       all_ref_key,
                       master_structure_info_dict, 
                       biopsy_z_voxel_length, 
                       num_dose_calc_NN,
                       num_dose_NN_to_show_for_animation_plotting,
                       dose_views_jsons_paths_list,
                       containment_views_jsons_paths_list,
                       show_NN_dose_demonstration_plots,
                       show_NN_dose_demonstration_plots_all_trials_at_once,
                       show_num_containment_demonstration_plots,
                       containment_results_structure_types_to_show_per_trial,
                       plot_nearest_neighbour_surface_boundary_demonstration,
                       plot_relative_structure_centroid_demonstration,
                       biopsy_needle_compartment_length,
                       simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                       plot_uniform_shifts_to_check_plotly,
                       differential_dvh_resolution,
                       cumulative_dvh_resolution,
                       v_percent_DVH_to_calc_list,
                       volume_DVH_quantiles_to_calculate,
                       plot_translation_vectors_pointclouds,
                       plot_cupy_containment_distribution_results,
                       plot_shifted_biopsies,
                       structure_miss_probability_roi,
                       cancer_tissue_label,
                       default_exterior_tissue,
                       miss_structure_complement_label,
                       tissue_length_above_probability_threshold_list,
                       n_bootstraps_for_tissue_length_above_threshold,
                       perform_mc_containment_sim,
                       perform_mc_dose_sim,
                       spinner_type,
                       cupy_array_upper_limit_NxN_size_input,
                       nearest_zslice_vals_and_indices_cupy_generic_max_size,
                       idw_power,
                       raw_data_mc_dosimetry_dump_bool, 
                       raw_data_mc_containment_dump_bool,
                       keep_light_containment_and_distances_to_relative_structures_dataframe_bool,
                       show_non_bx_relative_structure_z_dilation_bool,
                        show_non_bx_relative_structure_xy_dilation_bool,
                        generate_cuda_log_files,
                        custom_cuda_kernel_type
                       ):
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, completed_sections_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    with live_display:
        live_display.start(refresh = True)
        num_patients = master_structure_info_dict["Global"]["Num cases"]
        num_global_structures = master_structure_info_dict["Global"]["Num structures"]
        num_MC_dose_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
        num_MC_containment_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"]
        bx_sample_pt_lattice_spacing = master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing (mm)"]
        bx_sample_pts_volume_element = master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"]
        
        #live_display.stop()
        max_simulations = master_structure_info_dict["Global"]["MC info"]["Max of num MC simulations"]
        


        """
        default_output = "Initializing"
        processing_patients_task_main_description = "[red]Generating {} MC samples for {} structures [{}]...".format(max_simulations,num_global_structures,default_output)
        processing_patients_task_completed_main_description = "[green]Generating {} MC samples for {} structures".format(max_simulations,num_global_structures)
        processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total = num_patients)
        processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total = num_patients, visible = False)

        # simulate all structure shifts in parallel and update the master reference dict
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            processing_patients_task_main_description = "[red]Generating {} MC samples for {} structures [{}]...".format(max_simulations,num_global_structures,patientUID)
            patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
            if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
                #MC_simulator_shift_biopsy_structures_uniform_generator_parallel(parallel_pool, pydicom_item, structs_referenced_list, bx_ref, biopsy_needle_compartment_length, max_simulations)
                sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_shift_biopsy_structures_uniform_generator_cupy(pydicom_item, bx_ref, biopsy_needle_compartment_length, max_simulations)
                # update the patient dictionary
                for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
                    structure_type = generated_shifts_info_list[0]
                    specific_structure_index = generated_shifts_info_list[1]
                    specific_structure_structure_uniform_dist_shift_samples_arr = generated_shifts_info_list[2]
                    pydicom_item[structure_type][specific_structure_index]["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"] = specific_structure_structure_uniform_dist_shift_samples_arr
            
            sp_structure_normal_dist_shift_samples_and_structure_reference_list = cupy_functions.MC_simulator_shift_all_structures_generator_cupy(pydicom_item, structs_referenced_list, max_simulations)
            # update the patient dictionary
            for generated_shifts_info_list in sp_structure_normal_dist_shift_samples_and_structure_reference_list:
                structure_type = generated_shifts_info_list[0]
                specific_structure_index = generated_shifts_info_list[1]
                specific_structure_structure_normal_dist_shift_samples_arr = generated_shifts_info_list[2]
                pydicom_item[structure_type][specific_structure_index]["MC data: Generated normal dist random samples arr"] = cp.asnumpy(specific_structure_structure_normal_dist_shift_samples_arr)
            
            #MC_simulator_shift_all_structures_generator_parallel(parallel_pool, pydicom_item, structs_referenced_list, max_simulations)
            #master_structure_reference_dict[patientUID] = patient_dict_updated_with_all_structs_generated_norm_dist_translation_samples
            patients_progress.update(processing_patients_task, advance = 1)
            completed_progress.update(processing_patients_task_completed, advance = 1)
        patients_progress.update(processing_patients_task, visible = False, refresh = True)
        completed_progress.update(processing_patients_task_completed, visible = True, refresh = True)
        live_display.refresh()

        num_biopsies_global = master_structure_info_dict["Global"]["Num biopsies"]
        num_OARs_global = master_structure_info_dict["Global"]["Num OARs"]
        num_DILs_global = master_structure_info_dict["Global"]["Num DILs"]
        
        simulation_info_important_line_str = "Simulation data: # MC containment simulations = {} | # MC dose simulations = {} | # lattice spacing for BX cores (mm) = {} | # biopsies = {} | # anatomical structures = {} | # patients = {}.".format(str(num_MC_containment_simulations), str(num_MC_dose_simulations), str(bx_sample_pt_lattice_spacing), str(num_biopsies_global), str(num_global_structures-num_biopsies_global), str(num_patients))
        important_info.add_text_line(simulation_info_important_line_str, live_display)
        """

        

        """
        #live_display.stop()
        default_patientUID = "initializing"
        translating_patients_main_desc = "[red]Transforming anatomy [{}]...".format(default_patientUID)
        translating_patients_structures_task = patients_progress.add_task(translating_patients_main_desc, total=num_patients)
        translating_patients_structures_task_completed = completed_progress.add_task("[green]Transforming anatomy", total=num_patients, visible = False)
        # simulate every biopsy sequentially
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            translating_patients_main_desc = "[red]Transforming anatomy [{}]...".format(patientUID)
            patients_progress.update(translating_patients_structures_task, description = translating_patients_main_desc)

            # create a dictionary of all non bx structures
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
            
            # set structure type to BX 
            local_patient_num_biopsies = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            translating_bx_and_structure_relative_main_desc = "[cyan]~For each biopsy [{}]...".format("initializing")
            translating_biopsy_relative_to_structures_task = biopsies_progress.add_task(translating_bx_and_structure_relative_main_desc, total=local_patient_num_biopsies)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                num_sampled_sp_bx_pts = specific_bx_structure["Num sampled bx pts"]
                translating_bx_and_structure_relative_main_desc = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi)
                biopsies_progress.update(translating_biopsy_relative_to_structures_task, description = translating_bx_and_structure_relative_main_desc)
                
                indeterminate_sub_desc_bx_shift = "[cyan]~~Shifting biopsy structure (BX shift) [{}]".format(specific_bx_structure_roi)
                indeterminate_sub_bx_shift_task = indeterminate_progress_sub.add_task(indeterminate_sub_desc_bx_shift, total=None)
                bx_only_shifted_randomly_sampled_bx_pts_3Darr = cupy_functions.MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_cupy(specific_bx_structure, simulate_uniform_bx_shifts_due_to_bx_needle_compartment, plot_uniform_shifts_to_check_plotly, num_sampled_sp_bx_pts, max_simulations)
                indeterminate_progress_sub.update(indeterminate_sub_bx_shift_task, visible = False)
                
                indeterminate_sub_desc_bx_shift = "[cyan]~~Shifting biopsy structure (relative OAR and DIL shifts) [{}]".format(specific_bx_structure_roi)
                indeterminate_sub_bx_shift_task = indeterminate_progress_sub.add_task(indeterminate_sub_desc_bx_shift, total=None)
                structure_shifted_bx_data_dict = cupy_functions.MC_simulator_translate_sampled_bx_points_3darr_structure_only_shift_cupy(pydicom_item, structs_referenced_list, bx_only_shifted_randomly_sampled_bx_pts_3Darr, structure_organized_for_bx_data_blank_dict, max_simulations, num_sampled_sp_bx_pts)
                indeterminate_progress_sub.update(indeterminate_sub_bx_shift_task, visible = False)

                for relative_structure_key, cupy_array in structure_shifted_bx_data_dict.items():
                    structure_shifted_bx_data_dict[relative_structure_key] = cp.asnumpy(cupy_array)
                
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx only shifted 3darr"] = bx_only_shifted_randomly_sampled_bx_pts_3Darr
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx and structure shifted dict"] = structure_shifted_bx_data_dict

                if plot_shifted_biopsies == True:
                    for relative_structure_key, bx_and_structure_shifted_bx_pts_3darray in structure_shifted_bx_data_dict.items():
                       bx_only_shifted_randomly_sampled_bx_pts_2darr = np.reshape(cp.asnumpy(bx_only_shifted_randomly_sampled_bx_pts_3Darr),(-1,3))
                       bx_only_shifted_bx_pts_pcd = point_containment_tools.create_point_cloud(bx_only_shifted_randomly_sampled_bx_pts_2darr, color = np.array([0,1,0]), random_color = False)
                       bx_and_structure_shifted_bx_pts_2darr = np.reshape(bx_and_structure_shifted_bx_pts_3darray,(-1,3))
                       bx_and_structure_shifted_bx_pts_pcd = point_containment_tools.create_point_cloud(bx_and_structure_shifted_bx_pts_2darr, color = np.array([1,0,0]), random_color = False)
                       unshifted_bx_sampled_pts_pcd_copy = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                       unshifted_bx_sampled_pts_pcd_copy.paint_uniform_color(np.array([0,1,1]))
                       non_bx_structure_type = relative_structure_key[1]
                       structure_index = relative_structure_key[3]
                       non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                       non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                       non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))
                       plotting_funcs.plot_geometries(bx_and_structure_shifted_bx_pts_pcd,unshifted_bx_sampled_pts_pcd_copy,non_bx_struct_interpolated_pts_pcd, bx_only_shifted_bx_pts_pcd)
                       
                biopsies_progress.update(translating_biopsy_relative_to_structures_task, advance = 1)
            biopsies_progress.update(translating_biopsy_relative_to_structures_task, visible = False)
            
            patients_progress.update(translating_patients_structures_task, advance=1)
            completed_progress.update(translating_patients_structures_task_completed, advance=1)
        patients_progress.update(translating_patients_structures_task, visible=False)
        completed_progress.update(translating_patients_structures_task_completed, visible=True)
        live_display.refresh()


        #live_display.stop()
        if plot_translation_vectors_pointclouds == True:
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                for structs in structs_referenced_list:
                    for specific_structure_index, specific_structure in enumerate(pydicom_item[structs]):
                        if structs == bx_ref:
                            total_rigid_shift_vectors_arr = cp.asnumpy(specific_structure["MC data: Total rigid shift vectors arr"])
                            #randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                            plotting_funcs.plot_point_clouds(total_rigid_shift_vectors_arr)
                            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([total_rigid_shift_vectors_arr], title_text = str(patientUID) + str(specific_structure["ROI"]))
                        else: 
                            total_rigid_shift_vectors_arr = cp.asnumpy(specific_structure["MC data: Generated normal dist random samples arr"])
                            plotting_funcs.plot_point_clouds(total_rigid_shift_vectors_arr)
                            plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([total_rigid_shift_vectors_arr], title_text = str(patientUID) + str(specific_structure["ROI"]))
        """

        """
        testing_nominal_biopsy_containment_patient_task = patients_progress.add_task("[red]Testing nominal biopsy containment...", total=num_patients)
        testing_nominal_biopsy_containment_patient_task_completed = completed_progress.add_task("[green]Testing nominal biopsy containment", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_structure_type]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            patients_progress.update(testing_nominal_biopsy_containment_patient_task, description = "[red]Testing nominal biopsy containment...[{}]...".format(patientUID))
            bx_structure_type = bx_ref
            testing_biopsy_nominal_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_structure_type]):
                num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                bx_specific_biopsy_nominal_containment_desc = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi)
                biopsies_progress.update(testing_biopsy_nominal_containment_task, description = bx_specific_biopsy_nominal_containment_desc)
                
                unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                unshifted_bx_sampled_pts_arr_XY = unshifted_bx_sampled_pts_arr[:,0:2]
                unshifted_bx_sampled_pts_arr_Z = unshifted_bx_sampled_pts_arr[:,2]

                unshifted_bx_sampled_pts_arr_XY_interleaved_1darr = unshifted_bx_sampled_pts_arr_XY.flatten()
                unshifted_bx_sampled_pts_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(unshifted_bx_sampled_pts_arr_XY_interleaved_1darr)

                testing_each_non_bx_structure_nominal_containment_task = structures_progress.add_task("[cyan]~~For each non-BX structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                structure_shifted_bx_data_dict = master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: bx and structure shifted dict"] 
                nominal_containment_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                relative_structure_nominal_containment_results_data_frames_list = []
                for structure_info in nominal_containment_results_for_fixed_bx_dict.keys():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_refnum = structure_info[2]
                    structure_index = structure_info[3]
                    
                    structures_progress.update(testing_each_non_bx_structure_nominal_containment_task, description = "[cyan]~~For each non-BX structure [{}]...".format(structure_roi))

                    non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                    non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                    non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))
                    prostate_interslice_interpolation_information = master_structure_reference_dict[patientUID]['OAR ref'][0]["Inter-slice interpolation information"]
                    prostate_interpolated_pts_np_arr = prostate_interslice_interpolation_information.interpolated_pts_np_arr
                    prostate_interpolated_pts_pcd = point_containment_tools.create_point_cloud(prostate_interpolated_pts_np_arr, color = np.array([0,1,1]))
                    
                    interpolated_zvlas_list = non_bx_struct_interslice_interpolation_information.zslice_vals_after_interpolation_list
                    non_bx_struct_max_zval = max(interpolated_zvlas_list)
                    non_bx_struct_min_zval = min(interpolated_zvlas_list)
                    nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input(interpolated_zvlas_list,unshifted_bx_sampled_pts_arr_Z)

                    non_bx_struct_zslices_list = non_bx_struct_interslice_interpolation_information.interpolated_pts_list
                    non_bx_struct_zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in non_bx_struct_zslices_list]
                    non_bx_struct_zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(non_bx_struct_zslices_polygons_list))

                    nominal_containment_info_grand_cudf_dataframe = point_containment_tools.cuspatial_points_contained(non_bx_struct_zslices_polygons_cuspatial_geoseries,
                               unshifted_bx_sampled_pts_XY_cuspatial_geoseries_points, 
                               unshifted_bx_sampled_pts_arr, 
                               nearest_interpolated_zslice_index_array,
                               nearest_interpolated_zslice_vals_array,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval, 
                               num_sample_pts_in_bx,
                               None, 
                               structure_info,
                               None,
                               non_bx_struct_interpolated_pts_pcd,
                               plot_cupy_nominal_containment_results,
                               True
                               )

                    relative_structure_nominal_containment_results_data_frames_list.append(nominal_containment_info_grand_cudf_dataframe)
                    
                    if (show_nominal_containment_demonstration_plots == True) & (non_bx_structure_type == 'DIL ref'):
                        bx_test_pts_color_R = nominal_containment_info_grand_cudf_dataframe[nominal_containment_info_grand_cudf_dataframe["Trial num"] == "Nominal"]["Pt clr R"].to_numpy()
                        bx_test_pts_color_G = nominal_containment_info_grand_cudf_dataframe[nominal_containment_info_grand_cudf_dataframe["Trial num"] == "Nominal"]["Pt clr G"].to_numpy()
                        bx_test_pts_color_B = nominal_containment_info_grand_cudf_dataframe[nominal_containment_info_grand_cudf_dataframe["Trial num"] == "Nominal"]["Pt clr B"].to_numpy()
                        bx_test_pts_color_arr = np.empty([num_sample_pts_in_bx,3])
                        bx_test_pts_color_arr[:,0] = bx_test_pts_color_R
                        bx_test_pts_color_arr[:,1] = bx_test_pts_color_G
                        bx_test_pts_color_arr[:,2] = bx_test_pts_color_B
                        unshifted_bx_sampled_pts_containment_pcd = point_containment_tools.create_point_cloud_with_colors_array(unshifted_bx_sampled_pts_arr, bx_test_pts_color_arr)
                        plotting_funcs.plot_geometries(unshifted_bx_sampled_pts_containment_pcd, non_bx_struct_interpolated_pts_pcd, label='Unknown')
                            
                    structures_progress.update(testing_each_non_bx_structure_nominal_containment_task, advance=1)
                
                structures_progress.remove_task(testing_each_non_bx_structure_nominal_containment_task)
                
                # concatenate containment results into a single dataframe
                nominal_containment_info_grand_all_structures_cudf_dataframe = cudf.concat(relative_structure_nominal_containment_results_data_frames_list, ignore_index=True)
                
                # Update the master dictionary
                master_structure_reference_dict[patientUID][bx_structure_type][specific_bx_structure_index]["MC data: Nominal containment raw results dataframe"] = nominal_containment_info_grand_all_structures_cudf_dataframe

                biopsies_progress.update(testing_biopsy_nominal_containment_task, advance=1)
            biopsies_progress.remove_task(testing_biopsy_nominal_containment_task)

            patients_progress.update(testing_nominal_biopsy_containment_patient_task, advance = 1)
            completed_progress.update(testing_nominal_biopsy_containment_patient_task_completed, advance = 1)
        patients_progress.update(testing_nominal_biopsy_containment_patient_task, visible = False)
        completed_progress.update(testing_nominal_biopsy_containment_patient_task_completed, visible = True)
        live_display.refresh()
        """

        structure_specific_results_dict_empty = {"Total successes (containment) list": None, 
                                                 "Binomial estimator list": None, 
                                                 "Confidence interval 95 (containment) list": None, 
                                                 "Standard error (containment) list": None,
                                                 "Nominal containment list": None
                                                 }
        mutual_structure_specific_results_dict_empty = {"Total successes (containment) list": None, 
                                    "Binomial estimator list": None, 
                                    "Confidence interval 95 (containment) list": None, 
                                    "Standard error (containment) list": None,
                                    "Nominal containment list": None
                                    }


        live_display.stop()
        testing_biopsy_containment_patient_task = patients_progress.add_task("[red]Testing biopsy containment (cuspatial)...", total=num_patients)
        testing_biopsy_containment_patient_task_completed = completed_progress.add_task("[green]Testing biopsy containment (cuspatial)", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)          
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            patients_progress.update(testing_biopsy_containment_patient_task, description = "[red]Testing biopsy containment (cuspatial) [{}]...".format(patientUID))
            testing_biopsy_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                specific_bx_structure_refnum = specific_bx_structure["Ref #"]
                bx_sim_bool = specific_bx_structure['Simulated bool']
                bx_sim_type = specific_bx_structure['Simulated type']

                bx_specific_biopsy_containment_desc = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi)
                biopsies_progress.update(testing_biopsy_containment_task, description = bx_specific_biopsy_containment_desc)

                # For nominal position containment test
                unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                
                # paint the unshifted bx sampled points purple for later viewing
                unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))

                testing_each_non_bx_structure_containment_task = structures_progress.add_task("[cyan]~~For each non-BX structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                
                structure_shifted_bx_data_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx and structure shifted dict"] 


                biopsy_structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_bx_structure) 
                
                containment_info_grand_all_structures_pandas_dataframe = pandas.DataFrame()
                for structure_info,shifted_bx_data_3darr_cp in structure_shifted_bx_data_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_refnum = structure_info[2]
                    structure_index = structure_info[3]
                    shifted_bx_data_3darr = cp.asnumpy(shifted_bx_data_3darr_cp)
                    structures_progress.update(testing_each_non_bx_structure_containment_task, description = "[cyan]~~For each non-BX structure [{}]...".format(structure_roi))


                    #### IMPORTANT NOTICE FOR ADDING GENERALIZED TRANSFORMATIONS IN THE FUTURE!
                    ### PERFORM ALL TRANSFORMATIONS ON THE BIOPSY STRUCTURE IF POSSIBLE! 
                    ### NOTE: I HAVE INCLUDED Z AND RADIAL (XY) DILATIONS TO THE RELATIVE STRUCTURE ITSELF


                    # Extract and calcualte relative structure info
                    non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                    non_bx_struct_intraslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Intra-slice interpolation information"] # This is used for NN surface distance calculation!
                    non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                    non_bx_struct_interpolated_pts_with_endcaps_np_arr = non_bx_struct_intraslice_interpolation_information.interpolated_pts_with_end_caps_np_arr # This is used for NN surface distance calculation!
                    
                    # Don't use inter slice interpolated structures for prostate, it will be too slow
                    if (non_bx_structure_type == oar_ref) or (non_bx_structure_type == rectum_ref) or (non_bx_structure_type == urethra_ref):
                        non_bx_struct_zslices_list = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Equal num zslice contour pts"]
                        #non_bx_struct_org_config_zvals_list = polygon_dilation_helpers_numpy.extract_constant_z_values_single_configuration(non_bx_struct_zslices_list)
                    else: # If non prostate, urethra or rectum use the interpolated structures
                        non_bx_struct_zslices_list = non_bx_struct_interslice_interpolation_information.interpolated_pts_list
                        #non_bx_struct_org_config_zvals_list = non_bx_struct_interslice_interpolation_information.zslice_vals_after_interpolation_list
                    

                    # find max and min z values of relative structure
                    #non_bx_struct_max_zval = max(non_bx_struct_org_config_zvals_list)
                    #non_bx_struct_min_zval = min(non_bx_struct_org_config_zvals_list)


                    #non_bx_struct_zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in non_bx_struct_zslices_list]
                    #non_bx_struct_zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(non_bx_struct_zslices_polygons_list))




                    ### DILATION OF RELATIVE STRUCTURE (START)
                    # Dilate the structure for every trial
                    non_bx_structure_normal_dist_dilations_samples_arr = cp.asnumpy(pydicom_item[non_bx_structure_type][structure_index]["MC data: Generated normal dist random samples dilations arr"])

                    # This converts the structure from a list of constant z slice arrays to a 2d array with a partner indices array where each row is a zslice with two indices indicating the start and end index +1 of that slice
                    non_bx_struct_org_config_2d_arr, non_bx_struct_org_config_indices_slices_arr = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(non_bx_struct_zslices_list)

                    # Generate all trials dilated structures
                    #st = time.time()
                    dilated_structures_list, dilated_structures_slices_indices_list = polygon_dilation_helpers_numpy.generate_dilated_structures_parallelized(non_bx_struct_org_config_2d_arr, 
                                                                                                                                                              non_bx_struct_org_config_indices_slices_arr, 
                                                                                                                                                              non_bx_structure_normal_dist_dilations_samples_arr, 
                                                                                                                                                              show_non_bx_relative_structure_z_dilation_bool, 
                                                                                                                                                              show_non_bx_relative_structure_xy_dilation_bool, 
                                                                                                                                                              parallel_pool)
                    #et = time.time()
                    #print("Time to generate dilated structures: ", et-st)

                    # For each non dilated (original structure) z slices list, the polygons are NOT closed, ie. the last point is not the same as the first point. This needs to be corrected for the CONTAINMENT algorithm
                    # NOTE: This does not matter for the above generate_dilated_structures_parallelized function, because the generate_dilated_structures_parallelized function automatically returns closed polygons through shapely_polygon_anstance.exterior.coords method (see the polygon_dilation_helpers_numpy.dilate_polygons_xy_plane function)
                    non_bx_struct_zslices_list_closed_polygons = copy.deepcopy(non_bx_struct_zslices_list)
                    for i, zslice_arr in enumerate(non_bx_struct_zslices_list):
                        # append the first point to the end of the array
                        non_bx_struct_zslices_list_closed_polygons[i] = np.append(zslice_arr, zslice_arr[0][np.newaxis, :], axis=0)

                    # This converts the structure from a list of constant z slice arrays to a 2d array with a partner indices array where each row is a zslice with two indices indicating the start and end index +1 of that slice
                    non_bx_struct_org_config_2d_arr_closed_polygons, non_bx_struct_org_config_indices_slices_arr_closed_polygons = polygon_dilation_helpers_numpy.convert_to_2d_array_and_indices_numpy(non_bx_struct_zslices_list_closed_polygons)

                    # Prepend the nominal relative structure (closed polygons version)
                    nominal_and_dilated_structures_list_of_2d_arr = [non_bx_struct_org_config_2d_arr_closed_polygons] + dilated_structures_list
                    nominal_and_dilated_structures_slices_indices_list = [non_bx_struct_org_config_indices_slices_arr_closed_polygons] + dilated_structures_slices_indices_list

                    del dilated_structures_list # free up memory
                    del dilated_structures_slices_indices_list # free up memory

                    # Get the z values of nominal and all dilated slices for every trial
                    non_bx_struct_nominal_and_all_dilations_zvals_list = polygon_dilation_helpers_numpy.extract_constant_z_values(nominal_and_dilated_structures_list_of_2d_arr, nominal_and_dilated_structures_slices_indices_list)

                    # Extract all trials of the biopsy points
                    shifted_bx_data_3darr_num_MC_containment_sims_cutoff = shifted_bx_data_3darr[0:num_MC_containment_simulations]
                    
                    # Prepend the nominal biopsy position
                    combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff = np.concatenate([unshifted_bx_sampled_pts_arr[np.newaxis, :, :],shifted_bx_data_3darr_num_MC_containment_sims_cutoff], axis=0)


                    # Find the nearest z slices (this function is the best, it is very fast and checked that it produces the correct results)
                    # The output is a 3d array with the first dimension being the trial number, ie references the relative dilated structure, the second dimension being the biopsy point, ie every row is a biopsy point, and the third dimension having three values, the trial number, the relative dilated structure index, the closest z slice index, and the z value of the closest z slice 
                    #st = time.time()
                    ####
                    grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array = polygon_dilation_helpers_numpy.nearest_zslice_vals_and_indices_all_structures_3d_point_arr(non_bx_struct_nominal_and_all_dilations_zvals_list, 
                                                                                                                                                                              combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff)
                    ####
                    #et = time.time()
                    #print("Time to find nearest z slices (numpy): ", et-st)
                    


                    # Cupy function is slower!
                    """
                    st = time.time()
                    grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array_cupy = polygon_dilation_helpers_cupy.nearest_zslice_vals_and_indices_all_trials_cupy(non_bx_struct_nominal_and_all_dilations_zvals_list, combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff)
                    et = time.time()
                    print("Time to find nearest z slices (cupy): ", et-st)

                    # Test whether each arrays are equal
                    print("Are the nearest interpolated zslice index arrays equal? ", np.array_equal(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, cp.asnumpy(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array_cupy)))                    
                    """



                    # NOTE: For testing purposes only, the arrays should be equal, I am quite confident this old function worked so just wanted to check for consistency before switching to the new function polygon_dilation_helpers_numpy.nearest_zslice_vals_and_indices_all_trials
                    # Tested! They are equal!
                    """
                    st = time.time()
                    nominal_and_shifted_bx_data_3darr = np.concatenate([unshifted_bx_sampled_pts_arr[np.newaxis,:,:],shifted_bx_data_3darr],axis = 0)
                    
                    grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array = np.empty((nominal_and_shifted_bx_data_3darr.shape[0], nominal_and_shifted_bx_data_3darr.shape[1])) # length is number of sampled points * number of trials + 1 (for nominal)
                    grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array = np.empty((nominal_and_shifted_bx_data_3darr.shape[0], nominal_and_shifted_bx_data_3darr.shape[1]))
                    
                    for trial_index, sp_trial_dilated_structure_zvals_list in enumerate(non_bx_struct_nominal_and_all_dilations_zvals_list):
                        sp_trial_shifted_bx_data_2darr = nominal_and_shifted_bx_data_3darr[trial_index]
                        sp_trial_shifted_bx_data_2darr_Z = sp_trial_shifted_bx_data_2darr[:,2]

                        sp_trial_nearest_interpolated_zslice_index_array, sp_trial_nearest_interpolated_zslice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(sp_trial_dilated_structure_zvals_list, 
                                                                                                                                                            sp_trial_shifted_bx_data_2darr_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            structures_progress
                                                                                                                                                            )



                        grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array[trial_index,:] = sp_trial_nearest_interpolated_zslice_index_array
                        grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array[trial_index,:] = sp_trial_nearest_interpolated_zslice_vals_array


                    et = time.time()
                    print("Time to find nearest z-slices for all points and trials: ", et-st)


                    print("Are the nearest interpolated zslice index arrays equal? ", np.array_equal(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:,:,1], grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_array))
                    print("Are the nearest interpolated zslice locations arrays equal? ", np.array_equal(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array[:,:,2], grand_all_dilation_sp_trial_nearest_interpolated_zslice_vals_array))
                    """


                    ### DILATION OF RELATIVE STRUCTURE (END)





                    # Point clouds
                    non_bx_struct_interpolated_pts_pcd = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]['Interpolated structure point cloud dict']['Interslice']
                    #non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1])) # quicker to access from memory
                    
                    #prostate_interslice_interpolation_information = master_structure_reference_dict[patientUID]['OAR ref'][0]["Inter-slice interpolation information"]
                    #prostate_interpolated_pts_np_arr = prostate_interslice_interpolation_information.interpolated_pts_np_arr
                    #prostate_interpolated_pts_pcd = point_containment_tools.create_point_cloud(prostate_interpolated_pts_np_arr, color = np.array([0,1,1]))

                    # Shifted
                    shifted_bx_data_3darr_num_MC_containment_sims_cutoff = shifted_bx_data_3darr[0:num_MC_containment_simulations]
                    shifted_bx_data_stacked_2darr_from_all_trials_3darray = np.reshape(shifted_bx_data_3darr_num_MC_containment_sims_cutoff,(-1,3))

                    # Combine nominal and shifted
                    combined_nominal_and_shifted_bx_pts_2d_arr_XYZ = np.vstack((unshifted_bx_sampled_pts_arr,shifted_bx_data_stacked_2darr_from_all_trials_3darray))
                    combined_nominal_and_shifted_bx_pts_2d_arr_XY = combined_nominal_and_shifted_bx_pts_2d_arr_XYZ[:,0:2]
                    combined_nominal_and_shifted_bx_pts_2d_arr_Z = combined_nominal_and_shifted_bx_pts_2d_arr_XYZ[:,2]
                    del shifted_bx_data_stacked_2darr_from_all_trials_3darray



                    ### BEGIN DISTANCE CALCULATION SECTION TO RELATIVE STRUCTURE CENTROID!

                    # Calculate distances between sampled bx and every structure
                    ### IMPORTANT NOTICE WHEN GENERALIZING THIS IN THE FUTURE!
                    ### If you add any generalized transformations make sure that you are passing the completely transformed biopsy structure to the below distance calculation so that it accurately captures all transfoormations!
                    non_bx_structure_global_centroid = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Structure global centroid"].copy()
                    non_bx_structure_global_centroid_reshaped_for_broadcast = np.reshape(non_bx_structure_global_centroid,(1,1,3))
                    #combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff = np.concatenate([unshifted_bx_sampled_pts_arr[np.newaxis, :, :],shifted_bx_data_3darr_num_MC_containment_sims_cutoff], axis=0)
                    non_bx_structure_centroid_to_bx_points_vectors_all_trials = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff - non_bx_structure_global_centroid_reshaped_for_broadcast
                    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances = np.linalg.norm(non_bx_structure_centroid_to_bx_points_vectors_all_trials, axis=2)

                    # Flatten the distances
                    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened = non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances.flatten()

                    # Extract X, Y, Z coordinates and flatten them
                    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_X = non_bx_structure_centroid_to_bx_points_vectors_all_trials[:, :, 0].flatten()
                    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Y = non_bx_structure_centroid_to_bx_points_vectors_all_trials[:, :, 1].flatten()
                    non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Z = non_bx_structure_centroid_to_bx_points_vectors_all_trials[:, :, 2].flatten()
                    
                    structure_centroid_distances_df = pandas.DataFrame({"Trial num": np.repeat(np.arange(num_MC_containment_simulations+1), num_sample_pts_in_bx),
                                      "Original pt index": np.tile(np.arange(num_sample_pts_in_bx), num_MC_containment_simulations+1),
                                      'Dist. from struct. centroid': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened,
                                      'Dist. from struct. centroid X': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_X,
                                      'Dist. from struct. centroid Y': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Y,
                                      'Dist. from struct. centroid Z': non_bx_structure_centroid_to_bx_points_vectors_all_trials_distances_flattened_Z})
                    
                    # Demonstrate to ensure its working?
                    if plot_relative_structure_centroid_demonstration == True:
                        for trial in np.arange(0,num_MC_containment_simulations+1):
                            _ = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, np.tile(non_bx_structure_global_centroid,(num_sample_pts_in_bx,1)), combined_nominal_and_shifted_bx_pts_2d_arr_XYZ[num_sample_pts_in_bx*trial:num_sample_pts_in_bx*(trial+1)], 1, draw_lines = True)
                    
                    ### END DISTANCE CALC SECTION




                    # THIS LINE IS VERY SLOW
                    #combined_nominal_and_shifted_nearest_interpolated_zslice_index_array, combined_nominal_and_shifted_nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input(interpolated_zvlas_list,combined_nominal_and_shifted_bx_pts_2d_arr_Z)

                    ### NUMPY IS SLOWER!!
                    """
                    combined_nominal_and_shifted_nearest_interpolated_zslice_index_array, combined_nominal_and_shifted_nearest_interpolated_zslice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_numpy_generic(interpolated_zvlas_list, 
                                                                                                                                                            combined_nominal_and_shifted_bx_pts_2d_arr_Z,
                                                                                                                                                            numpy_array_upper_limit_NxN_size_input,
                                                                                                                                                            structures_progress
                                                                                                                                                            )
                    """
                    # THIS IS QUICKER! 
                    # Dont need this anymore!
                    """
                    combined_nominal_and_shifted_nearest_interpolated_zslice_index_array, combined_nominal_and_shifted_nearest_interpolated_zslice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvlas_list, 
                                                                                                                                                            combined_nominal_and_shifted_bx_pts_2d_arr_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            structures_progress
                                                                                                                                                            )
                    """

                    





                    ### BEGIN NEAREST NEIGHBOUR BOUNDARY SEARCH SECTION

                    # We only need to test points on the nearest slice! Then we need to check against the z extent projection at each end and take the smallest value of the three
                    non_bx_struct_whole_structure_KDtree = scipy.spatial.KDTree(non_bx_struct_interpolated_pts_with_endcaps_np_arr)
                    nearest_distance_to_structure_boundary, nearest_point_index_to_structure_boundary = non_bx_struct_whole_structure_KDtree.query(combined_nominal_and_shifted_bx_pts_2d_arr_XYZ, k=1)


                    nearest_neighbour_boundary_distances_df = pandas.DataFrame({"Trial num": np.repeat(np.arange(num_MC_containment_simulations+1), num_sample_pts_in_bx),
                                      "Original pt index": np.tile(np.arange(num_sample_pts_in_bx), num_MC_containment_simulations+1),
                                      "Struct. boundary NN dist.": nearest_distance_to_structure_boundary})
                    
                   

                    # Demonstrate to ensure its working?
                    if plot_nearest_neighbour_surface_boundary_demonstration == True:
                        for trial in np.arange(0,num_MC_containment_simulations+1):
                            non_bx_struct_fully_interpolated_with_end_caps_pts_pcd = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]['Interpolated structure point cloud dict']['Full with end caps']
                            #non_bx_struct_fully_interpolated_with_end_caps_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_with_endcaps_np_arr, color = np.array([0,0,1])) # quicker to access from memory
                            NN_pts_on_comparison_struct = non_bx_struct_interpolated_pts_with_endcaps_np_arr[nearest_point_index_to_structure_boundary[num_sample_pts_in_bx*trial:num_sample_pts_in_bx*(trial+1)]]
                            _ = plotting_funcs.dose_point_cloud_with_lines_only_for_animation(unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_fully_interpolated_with_end_caps_pts_pcd, NN_pts_on_comparison_struct, combined_nominal_and_shifted_bx_pts_2d_arr_XYZ[num_sample_pts_in_bx*trial:num_sample_pts_in_bx*(trial+1)], 1, draw_lines = True)

                    #### END Nearest neighbour boundary search section






                    combined_nominal_and_shifted_bx_data_XY_interleaved_1darr = combined_nominal_and_shifted_bx_pts_2d_arr_XY.flatten()
                    combined_nominal_and_shifted_bx_data_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(combined_nominal_and_shifted_bx_data_XY_interleaved_1darr)
                    del combined_nominal_and_shifted_bx_data_XY_interleaved_1darr
                    del combined_nominal_and_shifted_bx_pts_2d_arr_XY
                    del combined_nominal_and_shifted_bx_pts_2d_arr_Z
                    """
                    containment_info_grand_cudf_dataframe = point_containment_tools.cuspatial_points_contained(non_bx_struct_zslices_polygons_cuspatial_geoseries,
                               combined_nominal_and_shifted_bx_data_XY_cuspatial_geoseries_points, 
                               combined_nominal_and_shifted_bx_pts_2d_arr_XYZ, 
                               combined_nominal_and_shifted_nearest_interpolated_zslice_index_array,
                               combined_nominal_and_shifted_nearest_interpolated_zslice_vals_array,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval, 
                               num_sample_pts_in_bx,
                               num_MC_containment_simulations,
                               structure_info
                               )
                    """
                    """
                    containment_info_grand_pandas_dataframe, live_display = point_containment_tools.cuspatial_points_contained_mc_sim_cupy_pandas(non_bx_struct_zslices_polygons_cuspatial_geoseries,
                                                                            combined_nominal_and_shifted_bx_data_XY_cuspatial_geoseries_points, 
                                                                            combined_nominal_and_shifted_bx_pts_2d_arr_XYZ, 
                                                                            combined_nominal_and_shifted_nearest_interpolated_zslice_index_array,
                                                                            combined_nominal_and_shifted_nearest_interpolated_zslice_vals_array,
                                                                            non_bx_struct_max_zval,
                                                                            non_bx_struct_min_zval, 
                                                                            num_sample_pts_in_bx,
                                                                            num_MC_containment_simulations,
                                                                            patientUID,
                                                                            biopsy_structure_info,
                                                                            structure_info,
                                                                            layout_groups,
                                                                            live_display,
                                                                            biopsies_progress,
                                                                            indeterminate_progress_sub,
                                                                            upper_limit_size_input = cupy_array_upper_limit_NxN_size_input
                    )
                    """


                    # This function works one-to-one, that is every point should have a matching polygon to test against
                    """
                    pr = cProfile.Profile()
                    pr.enable()
                    containment_result_cp_arr_geoseries_version = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_geoseries_version(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                                                                                        combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                                                                                        nominal_and_dilated_structures_list_of_2d_arr, 
                                                                                                        nominal_and_dilated_structures_slices_indices_list)
                    pr.disable()

                    # Print profiling results
                    s = io.StringIO()
                    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                    ps.print_stats()
                    print(s.getvalue())
                    """

                    """
                    lp = LineProfiler()
                    lp.add_function(custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_3d_arr_version)
                    lp.add_function(custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.one_to_one_point_in_polygon_cupy_arr_version)

                    lp_wrapper = lp(custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_3d_arr_version)
                    """

                    #pr = cProfile.Profile()
                    #pr.enable()
                    
                    log_sub_dirs_list = [patientUID, specific_bx_structure_roi, non_bx_structure_type]
                    if generate_cuda_log_files == True:
                        custom_cuda_log_file_name = patientUID + "_" + specific_bx_structure_roi + "_" + non_bx_structure_type + "_N-" + str(num_MC_containment_simulations) + "_containment_log.txt"
                    else:
                        custom_cuda_log_file_name = None
                    
                    containment_result_cp_arr_cupy_arr_version = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_3d_arr_version(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                                                                                        combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                                                                                        nominal_and_dilated_structures_list_of_2d_arr, 
                                                                                                        nominal_and_dilated_structures_slices_indices_list,
                                                                                                        log_sub_dirs_list = log_sub_dirs_list, 
                                                                                                        log_file_name = custom_cuda_log_file_name,
                                                                                                        include_edges_in_log = False,
                                                                                                        kernel_type=custom_cuda_kernel_type)
                    containment_result_cp_arr_cupy_arr_version_2 = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_3d_arr_version(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                                                                                        combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                                                                                        nominal_and_dilated_structures_list_of_2d_arr, 
                                                                                                        nominal_and_dilated_structures_slices_indices_list,
                                                                                                        log_sub_dirs_list = log_sub_dirs_list, 
                                                                                                        log_file_name = custom_cuda_log_file_name,
                                                                                                        include_edges_in_log = False,
                                                                                                        kernel_type="one_to_one_pip_kernel_advanced_reparameterized_version")
                    """
                    lp.enable()  
                    containment_result_cp_arr_cupy_arr_version = lp_wrapper(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array,
                                                                            combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
                                                                            nominal_and_dilated_structures_list_of_2d_arr,
                                                                            nominal_and_dilated_structures_slices_indices_list,
                                                                            log_sub_dirs_list = log_sub_dirs_list,
                                                                            log_file_name = custom_cuda_log_file_name,
                                                                            include_edges_in_log = False,
                                                                            kernel_type="one_to_one_pip_kernel_advanced")
                    lp.disable()  
                    lp.print_stats()

                    lp.enable()  
                    containment_result_cp_arr_cupy_arr_version_2 = lp_wrapper(grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array,
                                                                            combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff,
                                                                            nominal_and_dilated_structures_list_of_2d_arr,
                                                                            nominal_and_dilated_structures_slices_indices_list,
                                                                            log_sub_dirs_list = log_sub_dirs_list,
                                                                            log_file_name = custom_cuda_log_file_name,
                                                                            include_edges_in_log = False,
                                                                            kernel_type="one_to_one_pip_kernel_advanced_reparameterized_version")
                    lp.disable()  
                    lp.print_stats()
                    """
                    #pr.disable()

                    # Print profiling results
                    #s = io.StringIO()
                    #ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
                    #ps.print_stats()
                    #print(s.getvalue())

                    # Check  if the two arrays from two different kernels are exactly equal
                    print(f"\n✅ Do Results Match?", cp.all(containment_result_cp_arr_cupy_arr_version == containment_result_cp_arr_cupy_arr_version_2))

                    # Check if the two arrays are exactly equal
                    #are_equal = cp.array_equal(containment_result_cp_arr_geoseries_version, containment_result_cp_arr_cupy_arr_version)
                    #print(are_equal)  # Output: True


                    # Build the dataframe from the containment results
                    biopsy_structure_info = {
                        "Structure ID": specific_bx_structure_roi,
                        "Dicom ref num": specific_bx_structure_refnum,
                        "Index number": specific_bx_structure_index
                    }

                    containment_info_grand_pandas_dataframe = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe(patientUID, 
                                                                                                                                                      biopsy_structure_info, 
                                                                                                                                                      structure_info, 
                                                                                                                                                      grand_all_dilations_sp_trial_nearest_interpolated_zslice_index_and_zval_array, 
                                                                                                                                                      combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff, 
                                                                                                                                                      containment_result_cp_arr_cupy_arr_version)

                    del combined_nominal_and_shifted_bx_data_XY_cuspatial_geoseries_points


                    ### ADD NEAREST NEIGHBOUR BOUNDARY DISTANCE TO DATAFRAME

                    containment_info_grand_pandas_dataframe = pandas.merge(containment_info_grand_pandas_dataframe, nearest_neighbour_boundary_distances_df, on=["Trial num", "Original pt index"], how='left')

                    # Now apply the sign of the value "Struct. boundary NN dist." based on "Pt contained bool"
                    containment_info_grand_pandas_dataframe['Struct. boundary NN dist.'] = np.where(
                        containment_info_grand_pandas_dataframe['Pt contained bool'],  # Condition
                        -abs(containment_info_grand_pandas_dataframe['Struct. boundary NN dist.']),  # If True (inside), make negative
                        abs(containment_info_grand_pandas_dataframe['Struct. boundary NN dist.'])   # If False (outside), make positive
                    )
                        
                    ### ADD DISTANCE INFORMATION TO DATAFRAME
                    
                    containment_info_grand_pandas_dataframe = pandas.merge(containment_info_grand_pandas_dataframe, structure_centroid_distances_df, on=["Trial num", "Original pt index"], how='left')

                    





                    # TAKE A DUMP?
                    if raw_data_mc_containment_dump_bool == True:
                        raw_mc_output_dir = master_structure_info_dict["Global"]["Raw MC output dir"]
                        containment_raw_results_csv_name = 'mc_raw_results_containment.csv'
                        containment_raw_results_csv = raw_mc_output_dir.joinpath(containment_raw_results_csv_name)
                        with open(containment_raw_results_csv, 'a') as temp_file_obj:
                            containment_info_grand_pandas_dataframe.to_csv(temp_file_obj, mode='a', index=False, header=temp_file_obj.tell()==0)







                    containment_info_grand_all_structures_pandas_dataframe = pandas.concat([containment_info_grand_all_structures_pandas_dataframe,containment_info_grand_pandas_dataframe], ignore_index=True)


                    ### PLOT RESULTS PER TRIAL AGAINST A SINGLE STRUCTURE
                    if (show_num_containment_demonstration_plots > 0) & (non_bx_structure_type in containment_results_structure_types_to_show_per_trial):
                        for trial_num in np.arange(0,show_num_containment_demonstration_plots):
                            single_trial_shifted_bx_data_arr = combined_nominal_and_shifted_bx_data_3darr_num_MC_containment_sims_cutoff[trial_num]
                            bx_test_pts_color_R = containment_info_grand_pandas_dataframe[containment_info_grand_pandas_dataframe["Trial num"] == trial_num]["Pt clr R"].to_numpy()
                            bx_test_pts_color_G = containment_info_grand_pandas_dataframe[containment_info_grand_pandas_dataframe["Trial num"] == trial_num]["Pt clr G"].to_numpy()
                            bx_test_pts_color_B = containment_info_grand_pandas_dataframe[containment_info_grand_pandas_dataframe["Trial num"] == trial_num]["Pt clr B"].to_numpy()
                            bx_test_pts_color_arr = np.empty([num_sample_pts_in_bx,3])
                            bx_test_pts_color_arr[:,0] = bx_test_pts_color_R
                            bx_test_pts_color_arr[:,1] = bx_test_pts_color_G
                            bx_test_pts_color_arr[:,2] = bx_test_pts_color_B
                            structure_and_bx_shifted_bx_pcd = point_containment_tools.create_point_cloud_with_colors_array(single_trial_shifted_bx_data_arr, bx_test_pts_color_arr)
                            
                            non_bx_struct_org_config_interpolated_pts_pcd = point_containment_tools.create_point_cloud(nominal_and_dilated_structures_list_of_2d_arr[0], color = np.array([1,0.65,0])) # paint original non bx structure in orange

                            non_bx_struct_dilated_interpolated_pts_pcd = point_containment_tools.create_point_cloud(nominal_and_dilated_structures_list_of_2d_arr[trial_num], color = np.array([0,0,1])) # paint dilated structure in blue

                            plotting_funcs.plot_geometries(structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_dilated_interpolated_pts_pcd, non_bx_struct_org_config_interpolated_pts_pcd, label='Unknown')
                            
                            #plotting_funcs.plot_geometries(structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd, label='Unknown')
                            #plotting_funcs.plot_two_views_side_by_side([structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd], containment_views_jsons_paths_list[0], [structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd], containment_views_jsons_paths_list[1])
                            #plotting_funcs.plot_two_views_side_by_side([structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd], containment_views_jsons_paths_list[2], [structure_and_bx_shifted_bx_pcd, unshifted_bx_sampled_pts_copy_pcd, non_bx_struct_interpolated_pts_pcd, prostate_interpolated_pts_pcd], containment_views_jsons_paths_list[3])
                    

                    ### PLOT ALL RESULTS AGAINST SINGLE STRUCTURE, NOTE: THIS ONLY SHOWS THE NON DILATED STRUCTURE VERSION!
                    if plot_cupy_containment_distribution_results == True:
                        grand_cudf_dataframe = containment_info_grand_pandas_dataframe

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

                        # Extract relative structure point cloud, note that it will NOT show the variations of the dilated structure!
                        non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                        non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                        non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1])) # paint tested structure in blue
                        
                        geom_list = [colored_bx_test_pts_pcd,non_bx_struct_interpolated_pts_pcd]
                        
                        # Extract all other structures
                        for other_struct_type in structs_referenced_list:
                            if other_struct_type == bx_ref:
                                continue
                            for sp_struct_ind, sp_struct in enumerate(master_structure_reference_dict[patientUID][other_struct_type]):
                                if other_struct_type == non_bx_structure_type:
                                    if structure_index == sp_struct_ind:
                                        continue
                                non_bx_unique_struct_interslice_interpolation_information = sp_struct["Inter-slice interpolation information"]
                                non_bx_unique_struct_interpolated_pts_np_arr = non_bx_unique_struct_interslice_interpolation_information.interpolated_pts_np_arr
                                non_bx_unique_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_unique_struct_interpolated_pts_np_arr, color = np.array([1,0.65,0])) # paint non tested structure in orange
                                geom_list.append(non_bx_unique_struct_interpolated_pts_pcd)
                        
                        #live_display.stop()
                        print(f"Patient: {patientUID}, Bx: {specific_bx_structure_roi}, Test struct: {structure_roi}")
                        stopwatch.stop()
                        plotting_funcs.plot_geometries(*geom_list, label='Unknown')
                        stopwatch.start()
                        #live_display.start()


                    # free up GPU memory
                    del containment_info_grand_pandas_dataframe

                    structures_progress.update(testing_each_non_bx_structure_containment_task, advance=1)

                structures_progress.remove_task(testing_each_non_bx_structure_containment_task)



                ###
                ### START COMPILING RESULTS (PER BIOPSY) ###
                ###

                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing independent probabilities", total = None)
                ###

                ### COMPUTE INDEPENDENT PROBABILITIES
                
                # convert from pandas to cudf
                ### COMMENTED OUT BECAUSE TRYING PANDAS INSTEAD
                #containment_info_grand_all_structures_cudf_dataframe = cudf.from_pandas(containment_info_grand_all_structures_pandas_dataframe)


                # compute independent probabilities
                # Shifted, note that the nominal position is indicated by Trial num = 0

                ### MEMORY ERROR (GRAPHICAL) HERE WHEN DOING N=10000 TRIALS, MAY WANT TO TRY PANDAS INSTEAD 
                """
                mc_compiled_results_for_fixed_bx_dataframe = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Trial num"] != 0)
                                                                    ][["Relative structure ROI","Relative structure type","Pt contained bool","Original pt index", "Relative structure index"]
                                                                      ].groupby(["Relative structure ROI","Relative structure type","Relative structure index","Original pt index"]).sum().sort_index().reset_index().rename(columns={"Pt contained bool": "Total successes"})
                """
                # OKAY TRYING IT!
                mc_compiled_results_for_fixed_bx_dataframe = containment_info_grand_all_structures_pandas_dataframe[(containment_info_grand_all_structures_pandas_dataframe["Trial num"] != 0)
                                                                    ][["Relative structure ROI","Relative structure type","Pt contained bool","Original pt index", "Relative structure index"]
                                                                      ].groupby(["Relative structure ROI","Relative structure type","Relative structure index","Original pt index"]).sum().sort_index().reset_index().rename(columns={"Pt contained bool": "Total successes"})

                # calculate binomial estimator
                mc_compiled_results_for_fixed_bx_dataframe["Binomial estimator"] = mc_compiled_results_for_fixed_bx_dataframe["Total successes"]/num_MC_containment_simulations
                
                ### MEMORY ERROR (GRAPHICAL) HERE WHEN DOING N=10000 TRIALS, MAY WANT TO TRY PANDAS INSTEAD 
                # Nominal, note that the nominal position is indicated by Trial num = 0
                """
                mc_compiled_results_for_fixed_bx_dataframe_nominal = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Trial num"] == 0)
                                                                    ][
                                                                        ["Relative structure ROI","Relative structure type","Relative structure index","Original pt index", "Pt contained bool"]
                                                                        ].reset_index(drop = True).rename(columns={"Pt contained bool": "Nominal"})
                """
                # OKAY TRYING IT! SEE ALSO LINE 652 and 674 BELOW, WE DONT CONVERT THE RESULTS DATAFRAME TO PANDAS ANYMORE!
                mc_compiled_results_for_fixed_bx_dataframe_nominal = containment_info_grand_all_structures_pandas_dataframe[(containment_info_grand_all_structures_pandas_dataframe["Trial num"] == 0)
                                                                    ][
                                                                        ["Relative structure ROI","Relative structure type","Relative structure index","Original pt index", "Pt contained bool"]
                                                                        ].reset_index(drop = True).rename(columns={"Pt contained bool": "Nominal"})


                # convert nominal column from bool to uint8
                mc_compiled_results_for_fixed_bx_dataframe_nominal = mc_compiled_results_for_fixed_bx_dataframe_nominal.astype({'Nominal': 'uint8'})              

                # merge nominal and all trials dataframes
                mc_compiled_results_for_fixed_bx_dataframe = mc_compiled_results_for_fixed_bx_dataframe.merge(mc_compiled_results_for_fixed_bx_dataframe_nominal, how='inner', on = ["Relative structure ROI","Relative structure type","Relative structure index","Original pt index"])
                
                # sort values
                mc_compiled_results_for_fixed_bx_dataframe = mc_compiled_results_for_fixed_bx_dataframe.sort_values(['Relative structure ROI',"Relative structure type",'Relative structure index', 'Original pt index'], ascending=[False,False, True,True]).reset_index(drop=True)


                


                # add back patient, bx id information to dataframe
                mc_compiled_results_for_fixed_bx_dataframe.insert(0, 'Simulated type', bx_sim_type)
                mc_compiled_results_for_fixed_bx_dataframe.insert(0, 'Simulated bool', bx_sim_bool)
                mc_compiled_results_for_fixed_bx_dataframe.insert(0, 'Bx index', specific_bx_structure_index)
                mc_compiled_results_for_fixed_bx_dataframe.insert(0, 'Bx refnum', str(specific_bx_structure_refnum))
                mc_compiled_results_for_fixed_bx_dataframe.insert(0, 'Bx ID', specific_bx_structure_roi)
                mc_compiled_results_for_fixed_bx_dataframe.insert(0, 'Patient ID', patientUID)

                ### HERE!!!!
                # to pandas
                #mc_compiled_results_for_fixed_bx_dataframe = mc_compiled_results_for_fixed_bx_dataframe.to_pandas()

                # add biopsy point location in the bx frame 

                bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                mc_compiled_results_for_fixed_bx_dataframe = misc_tools.include_vector_columns_in_dataframe(mc_compiled_results_for_fixed_bx_dataframe, 
                                                                                           bx_points_bx_coords_sys_arr, 
                                                                                           reference_column_name = 'Original pt index', 
                                                                                           new_column_name_x = "X (Bx frame)", 
                                                                                           new_column_name_y = "Y (Bx frame)", 
                                                                                           new_column_name_z = "Z (Bx frame)")
                
                # Calculate and add to dataframe the binom est standard error
                mc_compiled_results_for_fixed_bx_dataframe['Binom est STD err'] = mc_compiled_results_for_fixed_bx_dataframe.apply(lambda row: mf.binomial_se_estimator(row['Binomial estimator'], num_MC_containment_simulations, row['Total successes']), axis=1)
                CI_results = mc_compiled_results_for_fixed_bx_dataframe.apply(lambda row: mf.binomial_CI_estimator_general(row['Binomial estimator'], num_MC_containment_simulations, confidence_level = 0.95), axis=1)
                mc_compiled_results_for_fixed_bx_dataframe['CI lower vals'] = CI_results.apply(lambda x: x[0])
                mc_compiled_results_for_fixed_bx_dataframe['CI upper vals'] = CI_results.apply(lambda x: x[1])

                # Add voxel columns
                reference_dimension_col_name = "Z (Bx frame)"
                mc_compiled_results_for_fixed_bx_dataframe = dataframe_builders.add_voxel_columns_helper_func(mc_compiled_results_for_fixed_bx_dataframe, biopsy_z_voxel_length, reference_dimension_col_name)
                
                
                # HERE!!!!!!
                # save memory
                #del containment_info_grand_all_structures_cudf_dataframe
                del mc_compiled_results_for_fixed_bx_dataframe_nominal


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ### CALC SUM TO 1 PROBABILITIES

                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing sum-to-one probabilities", total = None)
                ###


                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = compute_sum_to_one_probabilities_by_tissue_heirarchy_with_default_tissue_for_all_false_and_nominal(containment_info_grand_all_structures_pandas_dataframe,
                                                        structs_referenced_dict,
                                                        default_exterior_tissue = default_exterior_tissue)

                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe["Binomial estimator"] = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe["Total successes"]/num_MC_containment_simulations                                        

                # convert nominal column from bool to uint8
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.astype({'Nominal': 'uint8'})  


                # add back patient, bx id information to dataframe
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.insert(0, 'Simulated type', bx_sim_type)
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.insert(0, 'Simulated bool', bx_sim_bool)
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.insert(0, 'Bx index', specific_bx_structure_index)
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.insert(0, 'Bx refnum', str(specific_bx_structure_refnum))
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.insert(0, 'Bx ID', specific_bx_structure_roi)
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.insert(0, 'Patient ID', patientUID)



                # add biopsy point location in the bx frame 

                bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = misc_tools.include_vector_columns_in_dataframe(mc_compiled_results_sum_to_one_for_fixed_bx_dataframe, 
                                                                                           bx_points_bx_coords_sys_arr, 
                                                                                           reference_column_name = 'Original pt index', 
                                                                                           new_column_name_x = "X (Bx frame)", 
                                                                                           new_column_name_y = "Y (Bx frame)", 
                                                                                           new_column_name_z = "Z (Bx frame)")
                
                # Calculate and add to dataframe the binom est standard error
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe['Binom est STD err'] = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.apply(lambda row: mf.binomial_se_estimator(row['Binomial estimator'], num_MC_containment_simulations, row['Total successes']), axis=1)
                CI_results = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe.apply(lambda row: mf.binomial_CI_estimator_general(row['Binomial estimator'], num_MC_containment_simulations, confidence_level = 0.95), axis=1)
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe['CI lower vals'] = CI_results.apply(lambda x: x[0])
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe['CI upper vals'] = CI_results.apply(lambda x: x[1])


                # Add voxel columns
                reference_dimension_col_name = "Z (Bx frame)"
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = dataframe_builders.add_voxel_columns_helper_func(mc_compiled_results_sum_to_one_for_fixed_bx_dataframe, biopsy_z_voxel_length, reference_dimension_col_name)



                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###

                #################



                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing mutual probabilities", total = None)
                ###

                ### COMPUTE MUTUAL PROBABILITIES
                MC_compiled_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                non_bx_structures_info_list = MC_compiled_results_for_fixed_bx_dict.keys()
                structure_info_combinations_list = [com for sub in range(1,3) for com in itertools.combinations(non_bx_structures_info_list , sub + 1)] # generates combinations from the unqie roi list 
                
                containment_info_grand_all_structures_pandas_dataframe_copy = copy.deepcopy(containment_info_grand_all_structures_pandas_dataframe)
                containment_info_grand_all_structures_pandas_dataframe_copy["(ID,type,index)"] = tuple(zip(containment_info_grand_all_structures_pandas_dataframe_copy["Relative structure ROI"], containment_info_grand_all_structures_pandas_dataframe_copy["Relative structure type"], containment_info_grand_all_structures_pandas_dataframe_copy["Relative structure index"]))
                bx_mutual_containment_by_org_pt_all_combos_dataframe = pandas.DataFrame()
                for structure_info_combination_tuple in structure_info_combinations_list:

                    
                    # This is needed to extract the correct relative structures information, this
                    # is better over & statement to prevent exceptional cases of 
                    # structures of differing type with the same name but with an index contained 
                    # in the accepted list from getting through. 
                    combo_list_for_dataframe_checker = [(x[0],x[1],x[3]) for x in structure_info_combination_tuple]


                    # Note needed to convert cudf dataframe to pandas dataframe since cudf dataframe groupby object has no method "all()"
                    bx_mutual_containment_sp_combo_by_org_pt_dataframe = containment_info_grand_all_structures_pandas_dataframe_copy[(containment_info_grand_all_structures_pandas_dataframe_copy["(ID,type,index)"].isin(combo_list_for_dataframe_checker))
                                                                        & (containment_info_grand_all_structures_pandas_dataframe_copy["Trial num"] != 0)].reset_index()[
                                                                            ["Pt contained bool","Original pt index","Trial num"]
                                                                            ].groupby(["Original pt index","Trial num"]).all().groupby(["Original pt index"]).sum().reset_index().rename(columns={"Pt contained bool": "Total successes"})
                    bx_mutual_containment_sp_combo_by_org_pt_dataframe["Binomial estimator"] = bx_mutual_containment_sp_combo_by_org_pt_dataframe["Total successes"]/num_MC_containment_simulations
                    
                    structure_id_combination = [x[0] for x in structure_info_combination_tuple]
                    structure_type_combination = [x[1] for x in structure_info_combination_tuple]
                    structure_index_combination = [x[3] for x in structure_info_combination_tuple]

                    bx_mutual_containment_sp_combo_by_org_pt_dataframe.insert(loc=0, column="Structure index combination", value=[tuple(structure_index_combination)]*num_sample_pts_in_bx)
                    bx_mutual_containment_sp_combo_by_org_pt_dataframe.insert(loc=0, column="Structure type combination", value=[tuple(structure_type_combination)]*num_sample_pts_in_bx)
                    bx_mutual_containment_sp_combo_by_org_pt_dataframe.insert(loc=0, column="Structure ID combination", value=[tuple(structure_id_combination)]*num_sample_pts_in_bx)


                    bx_mutual_containment_by_org_pt_all_combos_dataframe = pandas.concat([bx_mutual_containment_by_org_pt_all_combos_dataframe, bx_mutual_containment_sp_combo_by_org_pt_dataframe], ignore_index = True)
                    
                    
                    del bx_mutual_containment_sp_combo_by_org_pt_dataframe
                del containment_info_grand_all_structures_pandas_dataframe_copy
                

                ### DISTANCES TO BOUNDARY AND CENTROIDS COMPILE

                ### Compile distances to NN boundary and centroid dataframe

                # Global   
                global_distances_grand_all_structures_pandas_dataframe = containment_info_grand_all_structures_pandas_dataframe.groupby(['Patient ID', 'Bx ID', 'Biopsy refnum', 'Bx index', 'Relative structure ROI', 'Relative structure type', 'Relative structure index'])[['Struct. boundary NN dist.', 'Dist. from struct. centroid', 'Dist. from struct. centroid X', 'Dist. from struct. centroid Y', 'Dist. from struct. centroid Z']].describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                
                global_distances_grand_all_structures_pandas_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
                global_distances_grand_all_structures_pandas_dataframe, 
                threshold=0.25
                )
                
                global_distances_grand_all_structures_pandas_dataframe.reset_index(inplace=True)
                
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim compiled distances global dataframe"] = global_distances_grand_all_structures_pandas_dataframe
                

                ### Point-wise and voxel-wise
                containment_info_grand_all_structures_pandas_dataframe_with_vector_cols = misc_tools.include_vector_columns_in_dataframe(containment_info_grand_all_structures_pandas_dataframe, 
                                                                                           bx_points_bx_coords_sys_arr, 
                                                                                           reference_column_name = 'Original pt index', 
                                                                                           new_column_name_x = "X (Bx frame)", 
                                                                                           new_column_name_y = "Y (Bx frame)", 
                                                                                           new_column_name_z = "Z (Bx frame)",
                                                                                           in_place = False)
                # Add voxel columns
                reference_dimension_col_name = "Z (Bx frame)"
                containment_info_grand_all_structures_pandas_dataframe_with_vector_and_voxel_cols = dataframe_builders.add_voxel_columns_helper_func(containment_info_grand_all_structures_pandas_dataframe_with_vector_cols, 
                                                                                                                              biopsy_z_voxel_length, 
                                                                                                                              reference_dimension_col_name, 
                                                                                                                              in_place = False)
                
                del containment_info_grand_all_structures_pandas_dataframe_with_vector_cols


                # Point wise
                distances_point_wise_grand_all_structures_pandas_dataframe = containment_info_grand_all_structures_pandas_dataframe_with_vector_and_voxel_cols.groupby(['Patient ID', 'Bx ID', 'Biopsy refnum', 'Bx index', 'Relative structure ROI', 'Relative structure type', 'Relative structure index', 'Original pt index',"X (Bx frame)","Y (Bx frame)","Z (Bx frame)", 'Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'])[['Struct. boundary NN dist.', 'Dist. from struct. centroid', 'Dist. from struct. centroid X', 'Dist. from struct. centroid Y', 'Dist. from struct. centroid Z']].describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                distances_point_wise_grand_all_structures_pandas_dataframe.reset_index(inplace=True)

                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim compiled distances point-wise dataframe"] = distances_point_wise_grand_all_structures_pandas_dataframe
                
                # Voxel wise
                distances_voxel_wise_grand_all_structures_pandas_dataframe = containment_info_grand_all_structures_pandas_dataframe_with_vector_and_voxel_cols.groupby(['Patient ID', 'Bx ID', 'Biopsy refnum', 'Bx index', 'Relative structure ROI', 'Relative structure type', 'Relative structure index', 'Voxel index', 'Voxel begin (Z)', 'Voxel end (Z)'])[['Struct. boundary NN dist.', 'Dist. from struct. centroid', 'Dist. from struct. centroid X', 'Dist. from struct. centroid Y', 'Dist. from struct. centroid Z']].describe(percentiles=[0.05,0.25,0.5,0.75,0.95])
                distances_voxel_wise_grand_all_structures_pandas_dataframe.reset_index(inplace=True)

                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim compiled distances voxel-wise dataframe"] = distances_voxel_wise_grand_all_structures_pandas_dataframe
                ### End point-wise and voxel-wise


                # Free up memory
                del containment_info_grand_all_structures_pandas_dataframe_with_vector_and_voxel_cols




                ### KEEP ENTIRE DATAFRAME? NOTE IF THERE ARE MEMRORY ISSUES CONSIDER REMOVING THIS DATAFRAME FROM STORAGE
                
                ### Keep lighter version of entire dataframe in case we want the distance to containment relationship for every trial
                if keep_light_containment_and_distances_to_relative_structures_dataframe_bool == True:
                    containment_info_grand_all_structures_pandas_dataframe_light = containment_info_grand_all_structures_pandas_dataframe.drop(columns=['Nearest zslice zval',  'Nearest zslice index',  'Pt clr R',  'Pt clr G',  'Pt clr B',  'Test pt X',  'Test pt Y',  'Test pt Z'])
                    containment_info_grand_all_structures_pandas_dataframe_light = dataframe_builders.convert_columns_to_categorical_and_downcast(
                    containment_info_grand_all_structures_pandas_dataframe_light, 
                    threshold=0.25
                    )
                
                
                    master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim containment and distance all trials dataframe (light)"] = containment_info_grand_all_structures_pandas_dataframe_light
                
            
                # Free up memory
                del containment_info_grand_all_structures_pandas_dataframe

                # Add x,y,z point coordinates
                bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                bx_mutual_containment_by_org_pt_all_combos_dataframe = misc_tools.include_vector_columns_in_dataframe(bx_mutual_containment_by_org_pt_all_combos_dataframe, 
                                                                                           bx_points_bx_coords_sys_arr, 
                                                                                           reference_column_name = 'Original pt index', 
                                                                                           new_column_name_x = "X (Bx frame)", 
                                                                                           new_column_name_y = "Y (Bx frame)", 
                                                                                           new_column_name_z = "Z (Bx frame)")
                

                # Add voxel columns
                reference_dimension_col_name = "Z (Bx frame)"
                bx_mutual_containment_by_org_pt_all_combos_dataframe = dataframe_builders.add_voxel_columns_helper_func(bx_mutual_containment_by_org_pt_all_combos_dataframe, biopsy_z_voxel_length, reference_dimension_col_name)
                

                # add back patient, bx id information to dataframe
                bx_mutual_containment_by_org_pt_all_combos_dataframe.insert(0, 'Simulated type', bx_sim_type)
                bx_mutual_containment_by_org_pt_all_combos_dataframe.insert(0, 'Simulated bool', bx_sim_bool)
                bx_mutual_containment_by_org_pt_all_combos_dataframe.insert(0, 'Bx index', specific_bx_structure_index)
                bx_mutual_containment_by_org_pt_all_combos_dataframe.insert(0, 'Bx refnum', str(specific_bx_structure_refnum))
                bx_mutual_containment_by_org_pt_all_combos_dataframe.insert(0, 'Bx ID', specific_bx_structure_roi)
                bx_mutual_containment_by_org_pt_all_combos_dataframe.insert(0, 'Patient ID', patientUID)
                
                #bx_mutual_containment_by_org_pt_all_combos_dataframe = bx_mutual_containment_by_org_pt_all_combos_dataframe.reset_index(drop = True)


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###


                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~This code should be phased out!", total = None)
                ###


                #### EVERNTUALLY WANT TO PHASE OUT THIS CODE!
                MC_compiled_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                for structure_info in MC_compiled_results_for_fixed_bx_dict.keys():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_index = structure_info[3]

                    structure_specific_results_dict = structure_specific_results_dict_empty.copy()

                    bx_containment_counter_by_org_pt_ind_list = mc_compiled_results_for_fixed_bx_dataframe[(mc_compiled_results_for_fixed_bx_dataframe["Relative structure ROI"] == structure_roi) &
                                                               (mc_compiled_results_for_fixed_bx_dataframe["Relative structure type"] == non_bx_structure_type) &
                                                               (mc_compiled_results_for_fixed_bx_dataframe["Relative structure index"] == structure_index)].sort_values(by=['Original pt index'])["Total successes"].to_list()
                    structure_specific_results_dict["Total successes (containment) list"] = bx_containment_counter_by_org_pt_ind_list 
                    
                                       
                    bx_containment_binomial_estimator_by_org_pt_ind_list = mc_compiled_results_for_fixed_bx_dataframe[(mc_compiled_results_for_fixed_bx_dataframe["Relative structure ROI"] == structure_roi) &
                                                               (mc_compiled_results_for_fixed_bx_dataframe["Relative structure type"] == non_bx_structure_type) &
                                                               (mc_compiled_results_for_fixed_bx_dataframe["Relative structure index"] == structure_index)].sort_values(by=['Original pt index'])["Binomial estimator"].to_list()
                    structure_specific_results_dict["Binomial estimator list"] = bx_containment_binomial_estimator_by_org_pt_ind_list
                    
                    bx_nominal_containment_counter_by_org_pt_ind_list = mc_compiled_results_for_fixed_bx_dataframe[(mc_compiled_results_for_fixed_bx_dataframe["Relative structure ROI"] == structure_roi) &
                                                               (mc_compiled_results_for_fixed_bx_dataframe["Relative structure type"] == non_bx_structure_type) &
                                                               (mc_compiled_results_for_fixed_bx_dataframe["Relative structure index"] == structure_index)].sort_values(by=['Original pt index'])["Nominal"].to_list()
                    structure_specific_results_dict["Nominal containment list"] = bx_nominal_containment_counter_by_org_pt_ind_list        

                    MC_compiled_results_for_fixed_bx_dict[structure_info] = structure_specific_results_dict

                #live_display.stop()
                non_bx_structures_info_list = MC_compiled_results_for_fixed_bx_dict.keys()
                structure_info_combinations_list = [com for sub in range(1,3) for com in itertools.combinations(non_bx_structures_info_list , sub + 1)] # generates combinations from the unqie roi list 
                mutual_MC_compiled_results_for_fixed_bx_dict = {}
                for structure_info_combination_tuple in structure_info_combinations_list:
                    #comparison_set = {(x[0],x[1],x[3]) for x in structure_info_combination_tuple}
                    
                    structure_id_combination = tuple([x[0] for x in structure_info_combination_tuple])
                    structure_type_combination = tuple([x[1] for x in structure_info_combination_tuple])
                    structure_index_combination = tuple([x[3] for x in structure_info_combination_tuple])

                    combination_structure_specific_results_dict = mutual_structure_specific_results_dict_empty.copy()

                    bx_mutual_containment_counter_by_org_pt_ind_list = bx_mutual_containment_by_org_pt_all_combos_dataframe[(bx_mutual_containment_by_org_pt_all_combos_dataframe["Structure ID combination"] == structure_id_combination) &
                                                                                                                            (bx_mutual_containment_by_org_pt_all_combos_dataframe["Structure type combination"] == structure_type_combination) &
                                                                                                                            (bx_mutual_containment_by_org_pt_all_combos_dataframe["Structure index combination"] == structure_index_combination)].sort_values(by = 'Original pt index')['Total successes'].to_list()
                    combination_structure_specific_results_dict["Total successes (containment) list"] = bx_mutual_containment_counter_by_org_pt_ind_list
                    
                    bx_containment_combination_binomial_estimator_by_org_pt_ind_list = bx_mutual_containment_by_org_pt_all_combos_dataframe[(bx_mutual_containment_by_org_pt_all_combos_dataframe["Structure ID combination"] == structure_id_combination) &
                                                                                                                            (bx_mutual_containment_by_org_pt_all_combos_dataframe["Structure type combination"] == structure_type_combination) &
                                                                                                                            (bx_mutual_containment_by_org_pt_all_combos_dataframe["Structure index combination"] == structure_index_combination)].sort_values(by = 'Original pt index')['Binomial estimator'].to_list()
                    combination_structure_specific_results_dict["Binomial estimator list"] = bx_containment_combination_binomial_estimator_by_org_pt_ind_list

                    mutual_MC_compiled_results_for_fixed_bx_dict[structure_info_combination_tuple] = combination_structure_specific_results_dict
                #live_display.start()
                #### EVERNTUALLY WANT TO PHASE OUT THIS CODE!
                

                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###                
                
                #live_display.stop()

                


                # Update the master dictionary
                mc_compiled_results_for_fixed_bx_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(mc_compiled_results_for_fixed_bx_dataframe, threshold=0.25)
                mc_compiled_results_sum_to_one_for_fixed_bx_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(mc_compiled_results_sum_to_one_for_fixed_bx_dataframe, threshold=0.25)
                bx_mutual_containment_by_org_pt_all_combos_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(bx_mutual_containment_by_org_pt_all_combos_dataframe, threshold=0.25)
                #live_display.start()
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim results dataframe"] = mc_compiled_results_for_fixed_bx_dataframe
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim sum-to-one results dataframe"] = mc_compiled_results_sum_to_one_for_fixed_bx_dataframe
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: mutual compiled sim results dataframe"] = bx_mutual_containment_by_org_pt_all_combos_dataframe

                
                ### EVENTUALLY WANT TO DELETE THIS CODE!
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim results"] = MC_compiled_results_for_fixed_bx_dict
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: mutual compiled sim results"] = mutual_MC_compiled_results_for_fixed_bx_dict
                ### EVENTUALLY WANT TO DELETE THIS CODE!
                
                
                
                #master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim containment raw results dataframe"] = containment_info_grand_all_structures_pandas_dataframe

                biopsies_progress.update(testing_biopsy_containment_task, advance=1)
            biopsies_progress.remove_task(testing_biopsy_containment_task)

            patients_progress.update(testing_biopsy_containment_patient_task, advance = 1)
            completed_progress.update(testing_biopsy_containment_patient_task_completed, advance = 1)
        patients_progress.update(testing_biopsy_containment_patient_task, visible = False)
        completed_progress.update(testing_biopsy_containment_patient_task_completed, visible = True)
        live_display.refresh()
        
        #live_display.start()


        # This was removed in favour of doing one structure at a time since saving all the results from all patients 
        # and all structures into a single dataframe and saving it costed too much memory
        # its better to compile the result of a single cuspatial result right away and then delete it
        # due to decrease in memory consumption
        """
        #live_display.stop()
        if plot_cupy_containment_distribution_results == True:
            plotting_biopsy_containment_cuspatial_patient_task = patients_progress.add_task("[red]Plotting containment (cuspatial) results...", total=num_patients)
            plotting_biopsy_containment_cuspatial_patient_task_completed = completed_progress.add_task("[green]Plotting containment (cuspatial) results", total=num_patients, visible = False)
            for patientUID,pydicom_item in master_structure_reference_dict.items():
                structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
                patients_progress.update(testing_biopsy_containment_patient_task, description = "[red]Testing biopsy containment (cuspatial) [{}]...".format(patientUID))
                for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                    containment_info_grand_all_structures_cudf_dataframe = cudf.from_pandas(master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim containment raw results dataframe"])
                    for relative_structure_info in structure_organized_for_bx_data_blank_dict.keys():
                        structure_roi = relative_structure_info[0]
                        non_bx_structure_type = relative_structure_info[1]
                        structure_index = relative_structure_info[3]

                        grand_cudf_dataframe = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                        & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
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

                        # Extract relative structure point cloud
                        non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
                        non_bx_struct_interpolated_pts_np_arr = non_bx_struct_interslice_interpolation_information.interpolated_pts_np_arr
                        non_bx_struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(non_bx_struct_interpolated_pts_np_arr, color = np.array([0,0,1]))

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

        """


        """
        #live_display.stop()
        structure_specific_results_dict_empty = {"Total successes (containment) list": None, 
                                                 "Binomial estimator list": None, 
                                                 "Confidence interval 95 (containment) list": None, 
                                                 "Standard error (containment) list": None,
                                                 "Nominal containment list": None
                                                 }
        mutual_structure_specific_results_dict_empty = {"Total successes (containment) list": None, 
                                                 "Binomial estimator list": None, 
                                                 "Confidence interval 95 (containment) list": None, 
                                                 "Standard error (containment) list": None,
                                                 "Nominal containment list": None
                                                 }
        compiling_results_patient_containment_task = patients_progress.add_task("[red]Compiling MC results ...", total=num_patients)
        compiling_results_patient_containment_task_completed = completed_progress.add_task("[green]Compiling MC results", total=num_patients, visible = False)  
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(compiling_results_patient_containment_task, description = "[red]Compiling MC results [{}]...".format(patientUID), total=num_patients)
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)           
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            compiling_results_biopsy_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(compiling_results_biopsy_containment_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                containment_info_grand_all_structures_cudf_dataframe = cudf.from_pandas(master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim containment raw results dataframe"])
                MC_compiled_results_for_fixed_bx_dict = structure_organized_for_bx_data_blank_dict.copy()
                
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
                sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
                sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs
                compiling_results_each_non_bx_structure_containment_task = structures_progress.add_task("[cyan]~~For each structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                for structure_info in MC_compiled_results_for_fixed_bx_dict.keys():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_index = structure_info[3]

                    structures_progress.update(compiling_results_each_non_bx_structure_containment_task, description = "[cyan]~~For each structure [{}]...".format(structure_roi), total=sp_patient_total_num_non_BXs)
                    
                    structure_specific_results_dict = structure_specific_results_dict_empty.copy()

                    # Much slower code commented out, list comprehension can be very slow!
                    #st = time.time()
                    #bx_containment_counter_by_org_pt_ind_list = [len(containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                                                           & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
                                                                                                           & (containment_info_grand_all_structures_cudf_dataframe["Pt contained bool"] == True)
                                                                                                           & (containment_info_grand_all_structures_cudf_dataframe["Original pt index"] == pt_index)
                                                                                                           ])
                                                                                                           for pt_index in range(num_sample_pts_in_bx)
                                                                                                           ]
                    #et = time.time()
                    #print("Org: "+str(et-st))
                    
                    
                    # compute independent probabilities
                    # Shifted, note that the nominal position is indicated by Trial num = 0
                    bx_containment_counter_by_org_pt_ind_list = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                        & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
                                                                        & (containment_info_grand_all_structures_cudf_dataframe["Trial num"] != 0)
                                                                        ].reset_index()[
                                                                            ["Pt contained bool","Original pt index"]
                                                                            ].groupby(["Original pt index"]).sum().sort_index().to_numpy().T.flatten(order = 'C').tolist()
                    structure_specific_results_dict["Total successes (containment) list"] = bx_containment_counter_by_org_pt_ind_list                    
                    bx_containment_binomial_estimator_by_org_pt_ind_list = [x/num_MC_containment_simulations for x in bx_containment_counter_by_org_pt_ind_list]
                    structure_specific_results_dict["Binomial estimator list"] = bx_containment_binomial_estimator_by_org_pt_ind_list

                    
                    
                    # Nominal, note that the nominal position is indicated by Trial num = 0
                    bx_nominal_containment_counter_by_org_pt_ind_list = containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                        & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
                                                                        & (containment_info_grand_all_structures_cudf_dataframe["Trial num"] == 0)
                                                                        ].reset_index()[
                                                                            ["Pt contained bool","Original pt index"]
                                                                            ].groupby(["Original pt index"]).sum().sort_index().to_numpy().T.flatten(order = 'C').tolist()
                    
                    structure_specific_results_dict["Nominal containment list"] = bx_nominal_containment_counter_by_org_pt_ind_list        

                    MC_compiled_results_for_fixed_bx_dict[structure_info] = structure_specific_results_dict

                    structures_progress.update(compiling_results_each_non_bx_structure_containment_task, advance=1)


                # compute mutual probabilities
                #unique_non_bx_structures_roi_list = containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"].unique() # generates a unique list of structure ROIs from the dataframe!
                #mutual_probabilities_dict = {}
                non_bx_structures_info_list = MC_compiled_results_for_fixed_bx_dict.keys()
                structure_info_combinations_list = [com for sub in range(1,3) for com in itertools.combinations(non_bx_structures_info_list , sub + 1)] # generates combinations from the unqie roi list 
                mutual_MC_compiled_results_for_fixed_bx_dict = {}
                for structure_info_combination_tuple in structure_info_combinations_list:
                    roi_combination_list = [struct_info[0] for struct_info in structure_info_combination_tuple]
                    struct_index_combination_list = [struct_info[3] for struct_info in structure_info_combination_tuple]
                    
                    combination_structure_specific_results_dict = mutual_structure_specific_results_dict_empty.copy()

                    # Note needed to convert cudf dataframe to pandas dataframe since cudf dataframe groupby object has no method "all()"
                    containment_info_grand_all_structures_pandas_dataframe = containment_info_grand_all_structures_cudf_dataframe.to_pandas()
                    bx_mutual_containment_counter_by_org_pt_ind_list = containment_info_grand_all_structures_pandas_dataframe[(containment_info_grand_all_structures_pandas_dataframe["Relative structure ROI"].isin(roi_combination_list))  
                                                                        & (containment_info_grand_all_structures_pandas_dataframe["Relative structure index"].isin(struct_index_combination_list))
                                                                        ].reset_index()[
                                                                            ["Pt contained bool","Original pt index","Trial num"]
                                                                            ].groupby(["Original pt index","Trial num"]).all().sort_index().groupby(["Original pt index"]).sum().sort_index().to_numpy().T.flatten(order = 'C').tolist()
                    
                    combination_structure_specific_results_dict["Total successes (containment) list"] = bx_mutual_containment_counter_by_org_pt_ind_list
                    bx_containment_combination_binomial_estimator_by_org_pt_ind_list = [x/num_MC_containment_simulations for x in bx_mutual_containment_counter_by_org_pt_ind_list]
                    combination_structure_specific_results_dict["Binomial estimator list"] = bx_containment_combination_binomial_estimator_by_org_pt_ind_list

                    mutual_MC_compiled_results_for_fixed_bx_dict[structure_info_combination_tuple] = combination_structure_specific_results_dict
                
                del containment_info_grand_all_structures_cudf_dataframe
                del containment_info_grand_all_structures_pandas_dataframe
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim containment raw results dataframe"] = 'Deleted'

                structures_progress.remove_task(compiling_results_each_non_bx_structure_containment_task)
                biopsies_progress.update(compiling_results_biopsy_containment_task, advance = 1) 
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim results"] = MC_compiled_results_for_fixed_bx_dict
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: mutual compiled sim results"] = mutual_MC_compiled_results_for_fixed_bx_dict
            biopsies_progress.remove_task(compiling_results_biopsy_containment_task) 
            patients_progress.update(compiling_results_patient_containment_task, advance = 1) 
            completed_progress.update(compiling_results_patient_containment_task_completed, advance = 1)
        patients_progress.update(compiling_results_patient_containment_task, visible = False) 
        completed_progress.update(compiling_results_patient_containment_task_completed, visible = True)
        live_display.refresh()


        """

        
        calc_MC_stat_biopsy_containment_task = patients_progress.add_task("[red]Calculating MC statistics [{}]...".format("initializing"), total=num_patients)
        calc_MC_stat_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating MC statistics", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_MC_stat_biopsy_containment_task, description = "[red]Calculating MC statistics [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            calc_MC_stat_each_bx_structure_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_results_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim results"] 
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(calc_MC_stat_each_bx_structure_containment_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                for structureID,structure_specific_results_dict in specific_bx_results_dict.items():
                    bx_containment_binomial_estimator_by_org_pt_ind_list = structure_specific_results_dict["Binomial estimator list"]
                    bx_containment_counter_by_org_pt_ind_list = structure_specific_results_dict["Total successes (containment) list"] 
                    probability_estimator_list = bx_containment_binomial_estimator_by_org_pt_ind_list
                    num_successes_list = bx_containment_counter_by_org_pt_ind_list
                    num_trials = num_MC_containment_simulations
                    confidence_interval_list = calculate_binomial_containment_conf_intervals_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials)
                    standard_err_list = calculate_binomial_containment_stand_err_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials)
                    structure_specific_results_dict["Confidence interval 95 (containment) list"] = confidence_interval_list
                    structure_specific_results_dict["Standard error (containment) list"] = standard_err_list

                mutual_MC_compiled_results_for_fixed_bx_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: mutual compiled sim results"]
                for structureID,mutual_structure_specific_results_dict in mutual_MC_compiled_results_for_fixed_bx_dict.items():
                    bx_containment_binomial_estimator_by_org_pt_ind_list = mutual_structure_specific_results_dict["Binomial estimator list"]
                    bx_containment_counter_by_org_pt_ind_list = mutual_structure_specific_results_dict["Total successes (containment) list"] 
                    probability_estimator_list = bx_containment_binomial_estimator_by_org_pt_ind_list
                    num_successes_list = bx_containment_counter_by_org_pt_ind_list
                    num_trials = num_MC_containment_simulations
                    confidence_interval_list = calculate_binomial_containment_conf_intervals_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials)
                    standard_err_list = calculate_binomial_containment_stand_err_parallel(parallel_pool, probability_estimator_list, num_successes_list, num_trials)
                    mutual_structure_specific_results_dict["Confidence interval 95 (containment) list"] = confidence_interval_list
                    mutual_structure_specific_results_dict["Standard error (containment) list"] = standard_err_list

                    
                biopsies_progress.update(calc_MC_stat_each_bx_structure_containment_task, advance = 1)
            biopsies_progress.remove_task(calc_MC_stat_each_bx_structure_containment_task)
            patients_progress.update(calc_MC_stat_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_MC_stat_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_MC_stat_biopsy_containment_task, visible = False)
        completed_progress.update(calc_MC_stat_biopsy_containment_task_complete,visible = True)
        live_display.refresh()


        specific_bx_structure_tumor_tissue_dict_empty = {"Tumor tissue binomial est arr": None, 
                                            "Tumor tissue standard error arr": None, 
                                            "Tumor tissue confidence interval 95 arr": None, 
                                            "Tumor tissue nominal arr": None
                                            }
        
        specific_bx_structure_relative_OAR_miss_dict_empty = {"OAR miss structure info": None,
                            "OAR tissue miss binomial est arr": None,
                            "OAR tissue standard error arr": None, 
                            "OAR tissue miss confidence interval 95 2d arr": None,
                            "OAR tissue miss nominal arr": None
                            }
        
        calc_mutual_probabilities_stat_biopsy_containment_task = patients_progress.add_task("[red]Calculating mutual probabilities [{}]...".format("initializing"), total=num_patients)
        calc_mutual_probabilities_stat_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating mutual probabilities ", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task, description = "[red]Calculating mutual probabilities  [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            sp_patient_num_dils = master_structure_info_dict["By patient"][patientUID][dil_ref]["Num structs"]

            calc_mutual_probabilities_stat_each_bx_structure_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_in_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_tumor_tissue_dict = copy.deepcopy(specific_bx_structure_tumor_tissue_dict_empty)
                specific_bx_structure_relative_OAR_dict = copy.deepcopy(specific_bx_structure_relative_OAR_miss_dict_empty)

                specific_bx_results_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim results"] 
                mutual_MC_compiled_results_for_fixed_bx_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: mutual compiled sim results"]

                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(calc_mutual_probabilities_stat_each_bx_structure_containment_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
            
                ## calc probability tumor tissue
                # create an array to keep track of the sum of independent probabilities
                probability_sum_of_independent_pt_wise_dil_tissue_arr = np.empty((num_sample_pts_in_bx))
                probability_sum_of_independent_pt_wise_dil_tissue_arr.fill(0)

                # create a 2d array to accumulate the standard errors of each binomial estimator  
                mutual_MC_compiled_results_for_fixed_bx_exclusively_dils_only_dict = copy.deepcopy(mutual_MC_compiled_results_for_fixed_bx_dict)
                for mutual_struct_combo_key in mutual_MC_compiled_results_for_fixed_bx_dict.keys():
                    for struct_key in mutual_struct_combo_key:
                        struct_type = struct_key[1]
                        if struct_type != dil_ref:
                            mutual_MC_compiled_results_for_fixed_bx_exclusively_dils_only_dict.pop(mutual_struct_combo_key)
                            break
                num_mutual_combinations_involving_dils_only = len(mutual_MC_compiled_results_for_fixed_bx_exclusively_dils_only_dict)
                num_combinations_involving_dils_only = num_mutual_combinations_involving_dils_only + sp_patient_num_dils

                probabilities_standard_err_2d_arr = np.empty((num_combinations_involving_dils_only,num_sample_pts_in_bx))
                probabilities_standard_err_2d_arr.fill(0)

                # create nominal array for dil tissue
                bx_nominal_containment_counter_by_org_pt_ind_2d_arr = np.empty((sp_patient_num_dils,num_sample_pts_in_bx))

                # get independent probabilities and standard errors 
                index_for_se_arr = 0
                for structureID,structure_specific_results_dict in specific_bx_results_dict.items(): 
                    non_bx_structure_type = structureID[1]
                    # only accumulate dil probabilities
                    if non_bx_structure_type != dil_ref:
                        continue
                    specific_dil_structure_binomial_est_arr = np.array(structure_specific_results_dict["Binomial estimator list"])
                    probability_sum_of_independent_pt_wise_dil_tissue_arr = probability_sum_of_independent_pt_wise_dil_tissue_arr + specific_dil_structure_binomial_est_arr
                    
                    specific_dil_structure_binomial_se_est_arr = np.array(structure_specific_results_dict["Standard error (containment) list"])
                    probabilities_standard_err_2d_arr[index_for_se_arr,:] = specific_dil_structure_binomial_se_est_arr
                    
                    bx_nominal_containment_counter_by_org_pt_ind_arr = np.array(structure_specific_results_dict["Nominal containment list"])
                    bx_nominal_containment_counter_by_org_pt_ind_2d_arr[index_for_se_arr,:] = bx_nominal_containment_counter_by_org_pt_ind_arr

                    index_for_se_arr = index_for_se_arr + 1

                # calculate nominal containment 
                bx_nominal_containment_dil_exclusive_by_org_pt_ind_arr = np.any(bx_nominal_containment_counter_by_org_pt_ind_2d_arr, axis = 0)
                specific_bx_structure_tumor_tissue_dict["Tumor tissue nominal arr"] = bx_nominal_containment_dil_exclusive_by_org_pt_ind_arr.astype(int)

                # create an array to keep track of the sum of mutual probabilities
                probability_sum_of_mutual_pt_wise_dil_tissue_arr = np.empty((num_sample_pts_in_bx))
                probability_sum_of_mutual_pt_wise_dil_tissue_arr.fill(0)

                # get mutual probabilities and standard errors
                for mutual_structure_specific_results_dict in mutual_MC_compiled_results_for_fixed_bx_exclusively_dils_only_dict.values():
                    mutual_structure_specific_exlusively_dils_binomial_est_arr = np.array(mutual_structure_specific_results_dict["Binomial estimator list"])
                    probability_sum_of_mutual_pt_wise_dil_tissue_arr = probability_sum_of_mutual_pt_wise_dil_tissue_arr + mutual_structure_specific_exlusively_dils_binomial_est_arr
                    
                    specific_dil_structure_binomial_se_est_arr = np.array(structure_specific_results_dict["Standard error (containment) list"])
                    probabilities_standard_err_2d_arr[index_for_se_arr,:] = specific_dil_structure_binomial_se_est_arr
                    index_for_se_arr = index_for_se_arr + 1

                # create total probability pt wise array
                probability_pt_wise_dil_tissue_arr = np.empty((num_sample_pts_in_bx))
                probability_pt_wise_dil_tissue_arr.fill(0)

                probability_pt_wise_dil_tissue_arr = probability_sum_of_independent_pt_wise_dil_tissue_arr - probability_sum_of_mutual_pt_wise_dil_tissue_arr
                
                
                specific_bx_structure_tumor_tissue_dict["Tumor tissue binomial est arr"] = probability_pt_wise_dil_tissue_arr

                # calculate the standard error of the tumor tissue pt wise binomeial estimator
                probabilities_standard_err_arr = np.linalg.norm(probabilities_standard_err_2d_arr, axis=0, keepdims=False)
                specific_bx_structure_tumor_tissue_dict["Tumor tissue standard error arr"] = probabilities_standard_err_arr

                # calculate 95 CI 
                
                probabilities_CI_arr = mf.binomial_CI_estimator_general(probability_pt_wise_dil_tissue_arr, num_MC_containment_simulations, confidence_level = 0.95)
                #probabilities_CI_arr = mf.confidence_intervals_95_from_calculated_SE(probability_pt_wise_dil_tissue_arr, probabilities_standard_err_arr)
                specific_bx_structure_tumor_tissue_dict["Tumor tissue confidence interval 95 arr"] = probabilities_CI_arr


                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: tumor tissue probability"] = specific_bx_structure_tumor_tissue_dict


                
                

                # calculate miss probability
                for structure_info, structure_specific_results_dict in specific_bx_results_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    # only consider the miss structure
                    if (non_bx_structure_type != oar_ref) or (structure_roi != structure_miss_probability_roi):
                        continue
                    
                    non_bx_structure_binom_est_arr = np.array(structure_specific_results_dict["Binomial estimator list"])
                    miss_structure_binom_est_arr = 1 - non_bx_structure_binom_est_arr

                    miss_structure_standard_err_arr = np.array(structure_specific_results_dict["Standard error (containment) list"])
                    
                    non_bx_structure_CI_2d_arr = np.array(structure_specific_results_dict["Confidence interval 95 (containment) list"]).T
                    miss_structure_CI_2d_arr = 1 - non_bx_structure_CI_2d_arr
                    miss_structure_CI_2d_arr[[0,1]] = miss_structure_CI_2d_arr[[1,0]] # swap lower and upper rows as performing 1-A to get the complemeny swaps the upper and lower CIs

                    non_bx_structure_nominal_arr = np.array(structure_specific_results_dict["Nominal containment list"])
                    miss_structure_nominal_arr = 1- non_bx_structure_nominal_arr
                    
                    specific_bx_structure_relative_OAR_dict["OAR miss structure info"] = structure_info
                    specific_bx_structure_relative_OAR_dict["OAR tissue miss binomial est arr"] = miss_structure_binom_est_arr
                    specific_bx_structure_relative_OAR_dict["OAR tissue standard error arr"] = miss_structure_standard_err_arr
                    specific_bx_structure_relative_OAR_dict["OAR tissue miss confidence interval 95 2d arr"] = miss_structure_CI_2d_arr
                    specific_bx_structure_relative_OAR_dict["OAR tissue miss nominal arr"] = miss_structure_nominal_arr

                    break

                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: miss structure tissue probability"] = specific_bx_structure_relative_OAR_dict
                





                ### CREATE DATAFRAME OF MUTUAL PROBABILITIES
                containment_output_dict_by_MC_trial_for_pandas_data_frame, containment_output_by_MC_trial_pandas_data_frame = dataframe_builders.tissue_probability_dataframe_builder_by_bx_pt(patientUID,
                                                                                                                                                                                                specific_bx_structure,
                                                                                                                                                                                                specific_bx_structure_index, 
                                                                                                                                                                                                structure_miss_probability_roi,
                                                                                                                                                                                                cancer_tissue_label,
                                                                                                                                                                                                miss_structure_complement_label,
                                                                                                                                                                                                biopsy_z_voxel_length
                                                                                                                                                                                                )
                containment_output_by_MC_trial_pandas_data_frame = dataframe_builders.convert_columns_to_categorical_and_downcast(containment_output_by_MC_trial_pandas_data_frame, threshold=0.25)

                specific_bx_structure["Output data frames"]["Mutual containment output by bx point"] = containment_output_by_MC_trial_pandas_data_frame
                #specific_bx_structure["Output dicts for data frames"]["Mutual containment output by bx point"] = containment_output_dict_by_MC_trial_for_pandas_data_frame







                # calculate tissue volume above threshold 
                """
                bx_sample_pts_lattice_spacing = master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing (mm)"]
                
                bx_points_bx_coords_sys_arr = specific_bx_structure["Random uniformly sampled volume pts bx coord sys arr"]
                z_coords_arr = bx_points_bx_coords_sys_arr[:,2]
                """
                #live_display.stop()
                bx_sample_pts_volume_element = master_structure_info_dict["Global"]["MC info"]["BX sample pt volume element (mm^3)"]
                all_thresholds_volume_of_tissue_above_threshold_dataframe = pandas.DataFrame()
                for probability_threshold in tissue_length_above_probability_threshold_list:

                    volume_of_tissue_above_threshold_dataframe = tissue_volume_calculator(patientUID,
                             specific_bx_structure,
                             cancer_tissue_label,
                             probability_threshold,
                             bx_sample_pts_volume_element,
                             structure_miss_probability_roi)
                    
                    all_thresholds_volume_of_tissue_above_threshold_dataframe = pandas.concat([all_thresholds_volume_of_tissue_above_threshold_dataframe,volume_of_tissue_above_threshold_dataframe]).reset_index(drop = True)
                    """
                    length_estimate_distribution_arr, length_estimate_mean, length_estimate_se = tissue_length_calculator(z_coords_arr,
                                                                                                                        probability_pt_wise_dil_tissue_arr,
                                                                                                                        probabilities_standard_err_arr,
                                                                                                                        bx_sample_pts_lattice_spacing, 
                                                                                                                        threshold,
                                                                                                                        n_bootstraps_for_tissue_length_above_threshold)
                
                    tissue_length_by_threshold_dict[threshold] =  {"Length estimate distribution": length_estimate_distribution_arr,
                                                                                               "Num bootstraps": n_bootstraps_for_tissue_length_above_threshold,
                                                                                               "Length estimate mean": length_estimate_mean,
                                                                                               "Length estimate se": length_estimate_se}
                    """                            
                """
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: tissue length above threshold dict"] = tissue_length_by_threshold_dict 
                """
                #live_display.start()
                all_thresholds_volume_of_tissue_above_threshold_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(all_thresholds_volume_of_tissue_above_threshold_dataframe, threshold=0.25)

                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["Output data frames"]["Tissue volume above threshold"] = all_thresholds_volume_of_tissue_above_threshold_dataframe 








                biopsies_progress.update(calc_mutual_probabilities_stat_each_bx_structure_containment_task, advance = 1)
            biopsies_progress.remove_task(calc_mutual_probabilities_stat_each_bx_structure_containment_task)
            patients_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task, visible = False)
        completed_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task_complete,visible = True)
        live_display.refresh()


        """
        
        # voxelize containment results
        biopsy_voxelize_containment_task = patients_progress.add_task("[red]Voxelizing containment results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_containment_task_complete = completed_progress.add_task("[green]Voxelizing containment results", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing containment results [{}]...".format(patientUID))
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_containment_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_results_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: compiled sim results"] 
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_containment_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
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
                        total_num_MC_trials_in_voxel = num_sample_pts_in_voxel*num_MC_containment_simulations
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

        
        """



        #live_display.stop()
        ### NEW DOSIMETRIC LOCALIZATION!
        calc_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]Calculating NN dosimetric localization (NEW) [{}]...".format("initializing"), total=num_patients)
        calc_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating NN dosimetric localization (NEW)", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_dose_NN_biopsy_containment_task, description = "[red]Calculating NN dosimetric localization (NEW) [{}]...".format(patientUID))
            # create KDtree for dose data
            if dose_ref not in pydicom_item:
                patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
                completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
                continue

            dose_ref_dict = pydicom_item[dose_ref]
            phys_space_dose_map_and_gradient_map_3d_arr = dose_ref_dict["Dose and gradient phys space and pixel 3d arr"]
            phys_space_dose_map_3d_arr = phys_space_dose_map_and_gradient_map_3d_arr[:, :, :7]
            phys_space_dose_map_3d_arr_flattened = np.reshape(phys_space_dose_map_3d_arr, (-1,7) , order = 'C') # turn the data into a 2d array
            phys_space_dose_map_phys_coords_2d_arr = phys_space_dose_map_3d_arr_flattened[:,3:6] 
            phys_space_dose_map_dose_1d_arr = phys_space_dose_map_3d_arr_flattened[:,6] 
            dose_data_KDtree = scipy.spatial.KDTree(phys_space_dose_map_phys_coords_2d_arr)
            dose_ref_dict["KDtree"] = dose_data_KDtree
            

            # code for the plotting of the below NN search of sampled bx pts
            lattice_dose_pcd = dose_ref_dict["Dose grid point cloud"]
            thresholded_lattice_dose_pcd = dose_ref_dict["Dose grid point cloud thresholded"]

            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            dosimetric_calc_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(dosimetric_calc_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                


                bx_structure_info_dict = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_bx_structure)
                
                bx_only_shifted_3darr = cp.asnumpy(specific_bx_structure["MC data: bx only shifted 3darr"]) # note that the 3rd dimension slices are each MC trial
                bx_only_shifted_3darr_cutoff = bx_only_shifted_3darr[0:num_MC_dose_simulations]
                unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                unshifted_bx_sampled_pts_arr_3darr = np.expand_dims(unshifted_bx_sampled_pts_arr, axis=0)
                nominal_and_bx_only_shifted_3darr = np.concatenate((unshifted_bx_sampled_pts_arr_3darr,bx_only_shifted_3darr_cutoff))
                bx_only_shifted_stacked_2darr = np.reshape(nominal_and_bx_only_shifted_3darr, (-1,3) , order = 'C')

                
                dosimetric_calc_task = indeterminate_progress_sub.add_task("[cyan]~~Conducting NN search [{}]...".format(specific_bx_structure_roi), total = None)

                ### THIS DATAFRAME CONSUMES TOO MUCH MEMORY TO CARRY IT THROUGHOUT THE PROGRAMME, NEED TO PARSE IMMEDIATELY,
                ### CAN CONSIDER SAVING TO DISK... ONE STRATEGY COULD BE TO CONTINUALLY APPEND TO A CSV ON DISK!
                dose_nearest_neighbour_results_dataframe = dosimetric_localizer.dosimetric_localization_dataframe_version(bx_only_shifted_stacked_2darr,
                                                    patientUID, 
                                                    bx_structure_info_dict, 
                                                    dose_data_KDtree, 
                                                    phys_space_dose_map_dose_1d_arr, 
                                                    num_dose_calc_NN,
                                                    num_MC_dose_simulations,
                                                    num_sample_pts_per_bx,
                                                    idw_power)
                
                # TAKE A DUMP?
                if raw_data_mc_dosimetry_dump_bool == True:
                    raw_mc_output_dir = master_structure_info_dict["Global"]["Raw MC output dir"]
                    dose_raw_results_csv_name = 'mc_raw_results_dosimetry.csv'
                    dose_raw_results_csv = raw_mc_output_dir.joinpath(dose_raw_results_csv_name)
                    with open(dose_raw_results_csv, 'a') as temp_file_obj:
                        dose_nearest_neighbour_results_dataframe.to_csv(temp_file_obj, mode='a', index=False, header=temp_file_obj.tell()==0)

                indeterminate_progress_sub.remove_task(dosimetric_calc_task)

                # plot everything to make sure its working properly!
                if show_NN_dose_demonstration_plots == True:
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                    for trial_num in np.arange(0,num_MC_dose_simulations+1):
                        NN_pts_on_comparison_struct_for_all_points_concatenated = np.concatenate(dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Nearest phys space points"].to_numpy())
                        NN_doses_on_comparison_struct_for_all_points_concatenated = np.concatenate(dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Nearest doses"].to_numpy())
                        queried_bx_pts_arr_concatenated = np.stack(dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Struct test pt vec"].to_numpy())
                        queried_bx_pts_assigned_doses_arr_concatenated = dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Dose val (interpolated)"].to_numpy()
                        
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

                if show_NN_dose_demonstration_plots_all_trials_at_once == True:
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))

                    NN_pts_on_comparison_struct_for_all_points_concatenated = np.concatenate(dose_nearest_neighbour_results_dataframe["Nearest phys space points"].to_numpy())
                    queried_bx_pts_arr_concatenated = np.stack(dose_nearest_neighbour_results_dataframe["Struct test pt vec"].to_numpy())
                    
                    NN_doses_locations_pointcloud = point_containment_tools.create_point_cloud(NN_pts_on_comparison_struct_for_all_points_concatenated)
                    queried_bx_pts_locations_pointcloud = point_containment_tools.create_point_cloud(queried_bx_pts_arr_concatenated, color = np.array([0,1,0]))

                    pcd_list = [unshifted_bx_sampled_pts_copy_pcd, thresholded_lattice_dose_pcd, NN_doses_locations_pointcloud, queried_bx_pts_locations_pointcloud]
                    
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


                    del geometry_list_thresholded_dose_lattice
                else:
                    pass


                # Cant save these dataframes, they take up too much memory! Need to parse data right away
                #specific_bx_structure['MC data: bx to dose NN search results dataframe'] = dose_nearest_neighbour_results_dataframe # Note that trial 0 is the nominal position





                ### COMPILE DATA STRAIGHT AWAY!
                dose_nearest_neighbour_results_dataframe_pivoted = dose_nearest_neighbour_results_dataframe.pivot(index = "Original pt index", columns="Trial num", values = "Dose val (interpolated)")
                del dose_nearest_neighbour_results_dataframe
                
                # It seems pivoting already sorts the indices and columns, but just to be sure I do it manually anyways
                dose_nearest_neighbour_results_dataframe_pivoted_ensured_sorted = dose_nearest_neighbour_results_dataframe_pivoted.sort_index(axis = 0).sort_index(axis = 1)
                del dose_nearest_neighbour_results_dataframe_pivoted
                
                # Note that each row is a specific biopsy point, while the column is a particular MC trial
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = dose_nearest_neighbour_results_dataframe_pivoted_ensured_sorted.to_numpy()
                del dose_nearest_neighbour_results_dataframe_pivoted_ensured_sorted


                # Update master dictionary
                # MC trials only
                #specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (all MC trials)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,1:]
                # Nominal and MC trials
                #specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,0]
                specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr
                
                del dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr


                biopsies_progress.update(dosimetric_calc_biopsy_task, advance=1)
            biopsies_progress.remove_task(dosimetric_calc_biopsy_task)
            patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_dose_NN_biopsy_containment_task, visible = False)
        completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, visible = True)
        live_display.refresh()



        ### NEW DOSIMETRIC LOCALIZATION (GRADIENT)!
        calc_dose_NN_biopsy_containment_task = patients_progress.add_task("[red]Calculating NN dose gradient localization (NEW) [{}]...".format("initializing"), total=num_patients)
        calc_dose_NN_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating NN dose gradient localization (NEW)", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_dose_NN_biopsy_containment_task, description = "[red]Calculating NN dose gradient localization (NEW) [{}]...".format(patientUID))
            # create KDtree for dose data
            if dose_ref not in pydicom_item:
                patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
                completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
                continue

            dose_ref_dict = pydicom_item[dose_ref]
            phys_space_dose_map_and_gradient_map_3d_arr = dose_ref_dict["Dose and gradient phys space and pixel 3d arr"]
            phys_space_dose_gradient_map_3d_arr = np.delete(phys_space_dose_map_and_gradient_map_3d_arr, np.r_[6:10,11:14], axis=2)
            # note I am not going to rename all the variables to accomodate 'gradient' I have just done so for the first oen above, the rest is implied.
            phys_space_dose_map_3d_arr_flattened = np.reshape(phys_space_dose_gradient_map_3d_arr, (-1,7) , order = 'C') # turn the data into a 2d array
            phys_space_dose_map_phys_coords_2d_arr = phys_space_dose_map_3d_arr_flattened[:,3:6] 
            phys_space_dose_map_dose_1d_arr = phys_space_dose_map_3d_arr_flattened[:,6] 
            dose_data_KDtree = scipy.spatial.KDTree(phys_space_dose_map_phys_coords_2d_arr)
            dose_ref_dict["KDtree gradient"] = dose_data_KDtree
            

            # code for the plotting of the below NN search of sampled bx pts
            lattice_dose_pcd = dose_ref_dict["Dose grid gradient point cloud"]
            thresholded_lattice_dose_pcd = dose_ref_dict["Dose grid gradient point cloud thresholded"]

            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            dosimetric_calc_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(dosimetric_calc_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                


                bx_structure_info_dict = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_bx_structure)
                
                bx_only_shifted_3darr = cp.asnumpy(specific_bx_structure["MC data: bx only shifted 3darr"]) # note that the 3rd dimension slices are each MC trial
                bx_only_shifted_3darr_cutoff = bx_only_shifted_3darr[0:num_MC_dose_simulations]
                unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                unshifted_bx_sampled_pts_arr_3darr = np.expand_dims(unshifted_bx_sampled_pts_arr, axis=0)
                nominal_and_bx_only_shifted_3darr = np.concatenate((unshifted_bx_sampled_pts_arr_3darr,bx_only_shifted_3darr_cutoff))
                bx_only_shifted_stacked_2darr = np.reshape(nominal_and_bx_only_shifted_3darr, (-1,3) , order = 'C')

                
                dosimetric_calc_task = indeterminate_progress_sub.add_task("[cyan]~~Conducting NN search [{}]...".format(specific_bx_structure_roi), total = None)

                ### THIS DATAFRAME CONSUMES TOO MUCH MEMORY TO CARRY IT THROUGHOUT THE PROGRAMME, NEED TO PARSE IMMEDIATELY,
                ### CAN CONSIDER SAVING TO DISK... ONE STRATEGY COULD BE TO CONTINUALLY APPEND TO A CSV ON DISK!
                dose_grad_val_col_str = "Dose grad val (interpolated)"
                dose_nearest_neighbour_results_dataframe = dosimetric_localizer.dosimetric_localization_dataframe_version(bx_only_shifted_stacked_2darr,
                                                    patientUID, 
                                                    bx_structure_info_dict, 
                                                    dose_data_KDtree, 
                                                    phys_space_dose_map_dose_1d_arr, 
                                                    num_dose_calc_NN,
                                                    num_MC_dose_simulations,
                                                    num_sample_pts_per_bx,
                                                    idw_power,
                                                    result_col_name = "Dose grad val (interpolated)")
                
                # TAKE A DUMP?
                if raw_data_mc_dosimetry_dump_bool == True:
                    raw_mc_output_dir = master_structure_info_dict["Global"]["Raw MC output dir"]
                    dose_raw_results_csv_name = 'mc_raw_results_dosimetry_gradient.csv'
                    dose_raw_results_csv = raw_mc_output_dir.joinpath(dose_raw_results_csv_name)
                    with open(dose_raw_results_csv, 'a') as temp_file_obj:
                        dose_nearest_neighbour_results_dataframe.to_csv(temp_file_obj, mode='a', index=False, header=temp_file_obj.tell()==0)

                indeterminate_progress_sub.remove_task(dosimetric_calc_task)

                # plot everything to make sure its working properly!
                if show_NN_dose_demonstration_plots == True:
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                    for trial_num in np.arange(0,num_MC_dose_simulations+1):
                        NN_pts_on_comparison_struct_for_all_points_concatenated = np.concatenate(dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Nearest phys space points"].to_numpy())
                        NN_doses_on_comparison_struct_for_all_points_concatenated = np.concatenate(dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Nearest doses"].to_numpy())
                        queried_bx_pts_arr_concatenated = np.stack(dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num]["Struct test pt vec"].to_numpy())
                        queried_bx_pts_assigned_doses_arr_concatenated = dose_nearest_neighbour_results_dataframe[dose_nearest_neighbour_results_dataframe["Trial num"] == trial_num][dose_grad_val_col_str].to_numpy()
                        
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

                if show_NN_dose_demonstration_plots_all_trials_at_once == True:
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))

                    NN_pts_on_comparison_struct_for_all_points_concatenated = np.concatenate(dose_nearest_neighbour_results_dataframe["Nearest phys space points"].to_numpy())
                    queried_bx_pts_arr_concatenated = np.stack(dose_nearest_neighbour_results_dataframe["Struct test pt vec"].to_numpy())
                    
                    NN_doses_locations_pointcloud = point_containment_tools.create_point_cloud(NN_pts_on_comparison_struct_for_all_points_concatenated)
                    queried_bx_pts_locations_pointcloud = point_containment_tools.create_point_cloud(queried_bx_pts_arr_concatenated, color = np.array([0,1,0]))

                    pcd_list = [unshifted_bx_sampled_pts_copy_pcd, thresholded_lattice_dose_pcd, NN_doses_locations_pointcloud, queried_bx_pts_locations_pointcloud]
                    
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


                    del geometry_list_thresholded_dose_lattice
                else:
                    pass


                # Cant save these dataframes, they take up too much memory! Need to parse data right away
                #specific_bx_structure['MC data: bx to dose NN search results dataframe'] = dose_nearest_neighbour_results_dataframe # Note that trial 0 is the nominal position





                ### COMPILE DATA STRAIGHT AWAY!
                dose_nearest_neighbour_results_dataframe_pivoted = dose_nearest_neighbour_results_dataframe.pivot(index = "Original pt index", columns="Trial num", values = dose_grad_val_col_str)
                del dose_nearest_neighbour_results_dataframe
                
                # It seems pivoting already sorts the indices and columns, but just to be sure I do it manually anyways
                dose_nearest_neighbour_results_dataframe_pivoted_ensured_sorted = dose_nearest_neighbour_results_dataframe_pivoted.sort_index(axis = 0).sort_index(axis = 1)
                del dose_nearest_neighbour_results_dataframe_pivoted
                
                # Note that each row is a specific biopsy point, while the column is a particular MC trial
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = dose_nearest_neighbour_results_dataframe_pivoted_ensured_sorted.to_numpy()
                del dose_nearest_neighbour_results_dataframe_pivoted_ensured_sorted


                # Update master dictionary
                # MC trials only
                #specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (all MC trials)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,1:]
                # Nominal and MC trials
                #specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,0]
                specific_bx_structure["MC data: Dose gradient vals for each sampled bx pt arr (nominal & all MC trials)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr
                
                del dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr


                biopsies_progress.update(dosimetric_calc_biopsy_task, advance=1)
            biopsies_progress.remove_task(dosimetric_calc_biopsy_task)
            patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_dose_NN_biopsy_containment_task, visible = False)
        completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, visible = True)
        live_display.refresh()    



        #### OLD WAY, EVENTUALLY WANT TO PHASE OUT THIS CODE!
        """
        #live_display.stop()
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
            

            # code for the plotting of the below NN search of sampled bx pts
            lattice_dose_pcd = dose_ref_dict["Dose grid point cloud"]
            thresholded_lattice_dose_pcd = dose_ref_dict["Dose grid point cloud thresholded"]

            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            dosimetric_calc_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)           
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                num_sample_pts_per_bx = specific_bx_structure["Num sampled bx pts"]
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(dosimetric_calc_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                
                bx_only_shifted_3darr = cp.asnumpy(specific_bx_structure["MC data: bx only shifted 3darr"]) # note that the 3rd dimension slices are each MC trial
                bx_only_shifted_3darr_cutoff = bx_only_shifted_3darr[0:num_MC_dose_simulations]
                unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
                unshifted_bx_sampled_pts_arr_3darr = np.expand_dims(unshifted_bx_sampled_pts_arr, axis=0)
                nominal_and_bx_only_shifted_3darr = np.concatenate((unshifted_bx_sampled_pts_arr_3darr,bx_only_shifted_3darr_cutoff))

                
                dosimetric_calc_parallel_task = indeterminate_progress_sub.add_task("[cyan]~~Conducting NN search [{}]...".format(specific_bx_structure_roi), total = None)
                # non-parallel
                dosimetric_localization_all_MC_trials_list = dosimetric_localization_non_parallel(nominal_and_bx_only_shifted_3darr, 
                                                                                              specific_bx_structure, 
                                                                                              dose_ref_dict, 
                                                                                              dose_ref, 
                                                                                              phys_space_dose_map_phys_coords_2d_arr, 
                                                                                              phys_space_dose_map_dose_2d_arr, 
                                                                                              num_dose_calc_NN)
                
                # parallel MUCH SLOWER AND USES TOO MUCH MEMORY!
                
                #dosimetric_localization_all_MC_trials_list = dosimetric_localization_parallel(parallel_pool, 
                                                                                              nominal_and_bx_only_shifted_3darr, 
                                                                                              specific_bx_structure, 
                                                                                              dose_ref_dict, 
                                                                                              dose_ref, 
                                                                                              phys_space_dose_map_phys_coords_2d_arr, 
                                                                                              phys_space_dose_map_dose_2d_arr, 
                                                                                              num_dose_calc_NN)
                
                
                if show_NN_dose_demonstration_plots == True:
                    # plot everything to make sure its working properly!
                    unshifted_bx_sampled_pts_copy_pcd = copy.copy(specific_bx_structure['Random uniformly sampled volume pts pcd'])
                    unshifted_bx_sampled_pts_copy_pcd.paint_uniform_color(np.array([1,0,1]))
                    for mc_trial_NN_parent_dose_obj in dosimetric_localization_all_MC_trials_list:
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

                specific_bx_structure['MC data: bx to dose NN search objects list'] = dosimetric_localization_all_MC_trials_list # Note that entry 0 is the nominal position
                indeterminate_progress_sub.remove_task(dosimetric_calc_parallel_task)

                biopsies_progress.update(dosimetric_calc_biopsy_task, advance=1)
            biopsies_progress.remove_task(dosimetric_calc_biopsy_task)
            patients_progress.update(calc_dose_NN_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_dose_NN_biopsy_containment_task, visible = False)
        completed_progress.update(calc_dose_NN_biopsy_containment_task_complete, visible = True)
        live_display.refresh()
        #### OLD WAY, EVENTUALLY WANT TO PHASE OUT THIS CODE!
        """






        ### THIS WAS PHASED OUT IN FAVOUR OF COMPILING RIGHT AWAY TO SAVE MEMORY!
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
                dosimetric_localization_nominal_and_all_MC_trials_list = specific_bx_structure['MC data: bx to dose NN search objects list']
                
                # Split NN dose output into nominal and MC trials
                
                #dosimetric_localization_nominal_NN_parent_obj = dosimetric_localization_nominal_and_all_MC_trials_list[0]
                #dosimetric_localization_all_MC_trials_list = dosimetric_localization_nominal_and_all_MC_trials_list[1:]
                

                # MC trials
                dosimetric_localization_nominal_and_all_MC_trials_list_NN_lists_only = [NN_parent_obj.NN_data_list for NN_parent_obj in dosimetric_localization_nominal_and_all_MC_trials_list]
                dosimetric_localization_NN_child_objs_by_bx_point_nominal_and_all_trials_list = list(zip(*dosimetric_localization_nominal_and_all_MC_trials_list_NN_lists_only))
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_list = [[NN_child_obj.nearest_dose for NN_child_obj in fixed_bx_pt_NN_objs_list] for fixed_bx_pt_NN_objs_list in dosimetric_localization_NN_child_objs_by_bx_point_nominal_and_all_trials_list]
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = np.array(dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_list)
                # Nominal
                
                #dosimetric_localization_NN_child_objs_by_bx_point_nominal_list = dosimetric_localization_nominal_NN_parent_obj.NN_data_list
                #dosimetric_localization_dose_vals_by_bx_point_nominal_list = [nn_dose_child_obj.nearest_dose for nn_dose_child_obj in dosimetric_localization_NN_child_objs_by_bx_point_nominal_list]
                

                # Update master dictionary
                # MC trials only
                specific_bx_structure["MC data: Dose NN child obj for each sampled bx pt list (nominal & all MC trials)"] = dosimetric_localization_NN_child_objs_by_bx_point_nominal_and_all_trials_list
                specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (all MC trials)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,1:]
                # Nominal and MC trials
                #specific_bx_structure["MC data: Dose NN child obj for each sampled bx pt list (nominal)"] = dosimetric_localization_NN_child_objs_by_bx_point_nominal_and_all_trials_list[0]
                specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,0]
                specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"] = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr
                

                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, advance = 1)
            biopsies_progress.remove_task(compile_results_dose_NN_biopsy_containment_by_biopsy_task)    
            patients_progress.update(compile_results_dose_NN_biopsy_containment_task, advance=1)
            completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, advance=1)
        patients_progress.update(compile_results_dose_NN_biopsy_containment_task, visible = False)
        completed_progress.update(compile_results_dose_NN_biopsy_containment_task_complete, visible = True)
        live_display.refresh()


        """




        ##### NEW METHOD FOR CALCING DVH QUANTITIES


        #live_display.stop()
        calculate_biopsy_DVH_quantities_task = patients_progress.add_task("[red]Calculating DVH quantities [{}]...".format("initializing"), total=num_patients)
        calculate_biopsy_DVH_quantities_task_complete = completed_progress.add_task("[green]Calculating DVH quantities", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calculate_biopsy_DVH_quantities_task, description = "[red]Compiling dosimetric localization results [{}]...".format(patientUID))
            
            if dose_ref not in pydicom_item:
                patients_progress.update(calculate_biopsy_DVH_quantities_task, advance = 1)
                completed_progress.update(calculate_biopsy_DVH_quantities_task_complete, advance = 1)
                continue
            
            
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            calculate_biopsy_DVH_quantities_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)
            ctv_dose = pydicom_item[plan_ref]["Prescription doses dict"]["TARGET"]

            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(calculate_biopsy_DVH_quantities_by_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
                


                ### Prelims
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Prelims", total = None)
                ###

                # Extract data
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"]
                
                # Calc quantities
                max_dose_val_all_MC_trials = np.amax(dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr)
                minimum_dose_val = 0
                differential_dvh_range = (minimum_dose_val,max_dose_val_all_MC_trials)
                num_nominal_and_all_dose_trials = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr.shape[1]
                
                # Setting arrays
                differential_dvh_histogram_counts_by_MC_trial_arr = np.empty([num_nominal_and_all_dose_trials, differential_dvh_resolution]) # each row is a specific MC simulation, each column corresponds to the number of counts in that bin index 
                differential_dvh_histogram_edges_by_MC_trial_arr = np.empty([num_nominal_and_all_dose_trials, differential_dvh_resolution+1]) # each row is a specific MC simulation, each column corresponds to the bin edge 
                
                
                    
                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###


                ### Sp dose-dol metrics
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing histograms", total = None)
                ###
                    
                for trial_index in range(num_nominal_and_all_dose_trials):
                    # Extract particular trial dose vals
                    dosimetric_localization_dose_vals_all_pts_specific_MC_trial = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,trial_index]
                    
                    # Calc histogram and bin edges
                    # NUMPY HISTOGRAM DOESNT HAVE AN AXIS ARGUMENT! Could use .apply_along_axis but wont get much of a performance gain
                    differential_dvh_histogram_counts_specific_MC_trial, differential_dvh_histogram_edges_specific_MC_trial = np.histogram(dosimetric_localization_dose_vals_all_pts_specific_MC_trial, bins = differential_dvh_resolution, range = differential_dvh_range)
                    
                    # Save to preset arrays
                    differential_dvh_histogram_counts_by_MC_trial_arr[trial_index,:] = differential_dvh_histogram_counts_specific_MC_trial
                    differential_dvh_histogram_edges_by_MC_trial_arr[trial_index,:] = differential_dvh_histogram_edges_specific_MC_trial
                

                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###


                ### Sp dose-dol metrics
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing specific dose-vol metrics", total = None)
                ###
                
                # Create the template dictionary
                dvh_metric_vol_dose_percent_dict = {}
                for vol_dose_percent in v_percent_DVH_to_calc_list:
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)] = {"Nominal": None, 
                                                                               "All MC trials list": [], 
                                                                               "Mean": None, 
                                                                               "STD": None, 
                                                                               "Quantiles": None
                                                                               }
                
                
                # find specific DVH (V%) metrics for this particular trial
                for vol_dose_percent in v_percent_DVH_to_calc_list:
                    dose_threshold = (vol_dose_percent/100)*ctv_dose
                    truth_matrix_for_vol_dose_percent = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr > dose_threshold
                    counts_for_vol_dose_percent = np.sum(truth_matrix_for_vol_dose_percent, axis = 0)
                    percent_for_vol_dose_percent = (counts_for_vol_dose_percent/num_sampled_bx_pts)*100
                    
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Nominal"] = percent_for_vol_dose_percent[0]
                    
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["All MC trials list"] = percent_for_vol_dose_percent[1:].tolist()

                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ### 
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Stats of sp dose-vol metrics", total = None)
                ###

                for vol_dose_percent in v_percent_DVH_to_calc_list:
                    dvh_metric_vol_dose_percent_MC_trials_only_list = dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["All MC trials list"]
                    dvh_metric_all_trials_arr = np.array(dvh_metric_vol_dose_percent_MC_trials_only_list) 
                    mean_of_dvh_metric = np.mean(dvh_metric_all_trials_arr)
                    std_of_dvh_metric = np.std(dvh_metric_all_trials_arr)
                    quantiles_of_dvh_metric = {'Q'+str(q): np.quantile(dvh_metric_all_trials_arr, q/100) for q in volume_DVH_quantiles_to_calculate}
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Mean"] = mean_of_dvh_metric
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["STD"] = std_of_dvh_metric
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Quantiles"] = quantiles_of_dvh_metric


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###

                ### 
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Create DVH metric dataframe", total = None)
                ###

                bx_structure_info_dict = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_bx_structure)

                dvh_metric_dataframe_per_biopsy = pandas.DataFrame()
                for index, vol_dose_percent in enumerate(v_percent_DVH_to_calc_list):
                    dvh_metric_dict_for_dataframe_temp = {"Patient ID": patientUID,
                                                 "Bx ID": bx_structure_info_dict["Structure ID"],
                                                 "Struct type": bx_structure_info_dict["Struct ref type"],
                                                 "Dicom ref num": bx_structure_info_dict["Dicom ref num"],
                                                 "Bx index": bx_structure_info_dict["Index number"],
                                                 "DVH Metric": str(vol_dose_percent),
                                                 "Nominal": dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Nominal"],
                                                 "Mean": dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Mean"],
                                                 "Standard deviation": dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["STD"],
                                                }
                    
                    dvh_metric_dict_for_dataframe_temp.update(dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Quantiles"])
                    dvh_metric_dataframe_temp = pandas.DataFrame(dvh_metric_dict_for_dataframe_temp, index = [index])
                    del dvh_metric_dict_for_dataframe_temp                         

                    dvh_metric_dataframe_per_biopsy = pandas.concat([dvh_metric_dataframe_per_biopsy,dvh_metric_dataframe_temp], ignore_index=True)
                    del dvh_metric_dataframe_temp


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###




                ### 
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Differential and cumulative dvh", total = None)
                ###

                differential_dvh_histogram_volume_by_MC_trial_arr = differential_dvh_histogram_counts_by_MC_trial_arr*bx_sample_pts_volume_element
                differential_dvh_histogram_percent_by_MC_trial_arr = (differential_dvh_histogram_counts_by_MC_trial_arr/num_sampled_bx_pts)*100
                
                # Note that the nominal is excluded from the quantiles calculation
                differential_dvh_histogram_counts_quantiles_by_dose_bin = {'Q'+str(q): np.quantile(differential_dvh_histogram_counts_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                differential_dvh_histogram_volume_quantiles_by_dose_bin = {'Q'+str(q): np.quantile(differential_dvh_histogram_volume_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                differential_dvh_histogram_percent_quantiles_by_dose_bin = {'Q'+str(q): np.quantile(differential_dvh_histogram_percent_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}

                # Note that the 0th index is the nominal value for the counts, percent and volumes arrays
                differential_dvh_dict = {"Counts arr": differential_dvh_histogram_counts_by_MC_trial_arr, 
                                       "Percent arr": differential_dvh_histogram_percent_by_MC_trial_arr, 
                                       "Volume arr (cubic mm)": differential_dvh_histogram_volume_by_MC_trial_arr, 
                                       "Dose bins (edges) arr (Gy)": differential_dvh_histogram_edges_by_MC_trial_arr,
                                       "Quantiles counts dict": differential_dvh_histogram_counts_quantiles_by_dose_bin,
                                       "Quantiles percent dict": differential_dvh_histogram_percent_quantiles_by_dose_bin,
                                       "Quantiles volume dict": differential_dvh_histogram_volume_quantiles_by_dose_bin} # note that all rows in the edges array should be equal!

                # compute cumulative dvh quantities from differential dvh
                cumulative_dvh_counts_by_MC_trial_arr_D0_val = np.sum(differential_dvh_histogram_counts_by_MC_trial_arr, axis=1, keepdims = True)
                cumulative_dvh_counts_by_MC_trial_arr_Dgt0_vals = num_sampled_bx_pts - np.cumsum(differential_dvh_histogram_counts_by_MC_trial_arr, axis=1)
                cumulative_dvh_counts_by_MC_trial_arr = np.concatenate((cumulative_dvh_counts_by_MC_trial_arr_D0_val, cumulative_dvh_counts_by_MC_trial_arr_Dgt0_vals), axis=1)
                cumulative_dvh_volume_by_MC_trial_arr = cumulative_dvh_counts_by_MC_trial_arr*bx_sample_pts_volume_element
                cumulative_dvh_percent_by_MC_trial_arr = (cumulative_dvh_counts_by_MC_trial_arr/num_sampled_bx_pts)*100
                
                cumulative_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_histogram_edges_by_MC_trial_arr[0].copy()

                # Note that the nominal is excluded from the calculation
                cumulative_dvh_histogram_counts_quantiles_by_dose_val = {'Q'+str(q): np.quantile(cumulative_dvh_counts_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                cumulative_dvh_histogram_volume_quantiles_by_dose_val = {'Q'+str(q): np.quantile(cumulative_dvh_volume_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                cumulative_dvh_histogram_percent_quantiles_by_dose_val = {'Q'+str(q): np.quantile(cumulative_dvh_percent_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                
                # Note that the 0th index is the nominal value for the counts, percent and volumes arrays
                cumulative_dvh_dict = {"Counts arr": cumulative_dvh_counts_by_MC_trial_arr, 
                                       "Percent arr": cumulative_dvh_percent_by_MC_trial_arr, 
                                       "Volume arr (cubic mm)": cumulative_dvh_volume_by_MC_trial_arr, 
                                       "Dose vals arr (Gy)": cumulative_dvh_dose_vals_by_MC_trial_1darr,
                                       "Quantiles counts dict": cumulative_dvh_histogram_counts_quantiles_by_dose_val,
                                       "Quantiles percent dict": cumulative_dvh_histogram_percent_quantiles_by_dose_val,
                                       "Quantiles volume dict": cumulative_dvh_histogram_volume_quantiles_by_dose_val}
                
                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###

                specific_bx_structure["MC data: Differential DVH dict"] = differential_dvh_dict
                specific_bx_structure["MC data: Cumulative DVH dict"] = cumulative_dvh_dict 
                specific_bx_structure["MC data: dose volume metrics dict"] = dvh_metric_vol_dose_percent_dict 
                
                
                dvh_metric_dataframe_per_biopsy = dataframe_builders.convert_columns_to_categorical_and_downcast(dvh_metric_dataframe_per_biopsy, threshold=0.25)

                specific_bx_structure["Output data frames"]["DVH metrics"] = dvh_metric_dataframe_per_biopsy

                biopsies_progress.update(calculate_biopsy_DVH_quantities_by_biopsy_task, advance = 1)
            biopsies_progress.remove_task(calculate_biopsy_DVH_quantities_by_biopsy_task)    
            patients_progress.update(calculate_biopsy_DVH_quantities_task, advance=1)
            completed_progress.update(calculate_biopsy_DVH_quantities_task_complete, advance=1)
        patients_progress.update(calculate_biopsy_DVH_quantities_task, visible = False)
        completed_progress.update(calculate_biopsy_DVH_quantities_task_complete, visible = True)
        live_display.refresh()













        """
        #live_display.stop()
        calculate_biopsy_DVH_quantities_task = patients_progress.add_task("[red]Calculating DVH quantities [{}]...".format("initializing"), total=num_patients)
        calculate_biopsy_DVH_quantities_task_complete = completed_progress.add_task("[green]Calculating DVH quantities", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calculate_biopsy_DVH_quantities_task, description = "[red]Compiling dosimetric localization results [{}]...".format(patientUID))
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            calculate_biopsy_DVH_quantities_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)
            ctv_dose = pydicom_item[plan_ref]["Prescription doses dict"]["TARGET"]

            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(calculate_biopsy_DVH_quantities_by_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                num_sampled_bx_pts = specific_bx_structure["Num sampled bx pts"]
                
                ### Prelims
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Prelims", total = None)
                ###


                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"]
                max_dose_val_all_MC_trials = np.amax(dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr)
                minimum_dose_val = 0
                differential_dvh_range = (minimum_dose_val,max_dose_val_all_MC_trials)
                num_nominal_and_all_dose_trials = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr.shape[1]
                #differential_DVH_bin_length = (max_dose_val_all_MC_trials-minimum_dose_val)/differential_dvh_resolution
                #differential_DVH_bins_list = [[minimum_dose_val+j*differential_DVH_bin_length,minimum_dose_val+(j+1)*differential_DVH_bin_length]for j in range(differential_dvh_resolution)]
                differential_dvh_histogram_counts_by_MC_trial_arr = np.empty([num_nominal_and_all_dose_trials, differential_dvh_resolution]) # each row is a specific MC simulation, each column corresponds to the number of counts in that bin index 
                differential_dvh_histogram_edges_by_MC_trial_arr = np.empty([num_nominal_and_all_dose_trials, differential_dvh_resolution+1]) # each row is a specific MC simulation, each column corresponds to the bin edge 
                #cumulative_dvh_counts_by_MC_trial_arr = np.empty([num_MC_dose_simulations, differential_dvh_resolution+1]) # each row is a specific MC simulation, each column corresponds to the number of counts that satisfy the bound provided by the dose value bin edge of the same index of the differential_dvh_histogram_edges_by_MC_trial_arr
                dvh_metric_vol_dose_percent_dict = {}
                for vol_dose_percent in v_percent_DVH_to_calc_list:
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)] = {"Nominal": None, 
                                                                               "All MC trials list": [], 
                                                                               "Mean": None, 
                                                                               "STD": None, 
                                                                               "Quantiles": None
                                                                               }
                    
                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###


                ### Sp dose-fol metrics
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Specific dose-vol metrics", total = None)
                ###
                    
                for trial_index in range(num_nominal_and_all_dose_trials):
                    dosimetric_localization_dose_vals_all_pts_specific_MC_trial = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,trial_index]
                    differential_dvh_histogram_counts_specific_MC_trial, differential_dvh_histogram_edges_specific_MC_trial = np.histogram(dosimetric_localization_dose_vals_all_pts_specific_MC_trial, bins = differential_dvh_resolution, range = differential_dvh_range)
                    differential_dvh_histogram_counts_by_MC_trial_arr[trial_index,:] = differential_dvh_histogram_counts_specific_MC_trial
                    differential_dvh_histogram_edges_by_MC_trial_arr[trial_index,:] = differential_dvh_histogram_edges_specific_MC_trial
                    
                
                    # find specific DVH metrics for nominal and all MC trials
                    for vol_dose_percent in v_percent_DVH_to_calc_list:
                        dose_threshold = (vol_dose_percent/100)*ctv_dose
                        counts_for_vol_dose_percent = dosimetric_localization_dose_vals_all_pts_specific_MC_trial[dosimetric_localization_dose_vals_all_pts_specific_MC_trial > dose_threshold].shape[0]
                        percent_for_vol_dose_percent = (counts_for_vol_dose_percent/num_sampled_bx_pts)*100
                        if trial_index == 0:
                            dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Nominal"] = percent_for_vol_dose_percent
                        else:
                            dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["All MC trials list"].append(percent_for_vol_dose_percent)

                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ### 
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Stats of sp dose-vol metrics", total = None)
                ###

                for vol_dose_percent in v_percent_DVH_to_calc_list:
                    dvh_metric_vol_dose_percent_MC_trials_only_list = dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["All MC trials list"]
                    dvh_metric_all_trials_arr = np.array(dvh_metric_vol_dose_percent_MC_trials_only_list) 
                    mean_of_dvh_metric = np.mean(dvh_metric_all_trials_arr)
                    std_of_dvh_metric = np.std(dvh_metric_all_trials_arr)
                    quantiles_of_dvh_metric = {'Q'+str(q): np.quantile(dvh_metric_all_trials_arr, q/100) for q in volume_DVH_quantiles_to_calculate}
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Mean"] = mean_of_dvh_metric
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["STD"] = std_of_dvh_metric
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Quantiles"] = quantiles_of_dvh_metric


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ### 
                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Differential and cumulative dvh", total = None)
                ###

                differential_dvh_histogram_volume_by_MC_trial_arr = differential_dvh_histogram_counts_by_MC_trial_arr*bx_sample_pts_volume_element
                differential_dvh_histogram_percent_by_MC_trial_arr = (differential_dvh_histogram_counts_by_MC_trial_arr/num_sampled_bx_pts)*100
                
                # Note that the nominal is excluded from the quantiles calculation
                differential_dvh_histogram_counts_quantiles_by_dose_bin = {'Q'+str(q): np.quantile(differential_dvh_histogram_counts_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                differential_dvh_histogram_volume_quantiles_by_dose_bin = {'Q'+str(q): np.quantile(differential_dvh_histogram_volume_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                differential_dvh_histogram_percent_quantiles_by_dose_bin = {'Q'+str(q): np.quantile(differential_dvh_histogram_percent_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}

                # Note that the 0th index is the nominal value for the counts, percent and volumes arrays
                differential_dvh_dict = {"Counts arr": differential_dvh_histogram_counts_by_MC_trial_arr, 
                                       "Percent arr": differential_dvh_histogram_percent_by_MC_trial_arr, 
                                       "Volume arr (cubic mm)": differential_dvh_histogram_volume_by_MC_trial_arr, 
                                       "Dose bins (edges) arr (Gy)": differential_dvh_histogram_edges_by_MC_trial_arr,
                                       "Quantiles counts dict": differential_dvh_histogram_counts_quantiles_by_dose_bin,
                                       "Quantiles percent dict": differential_dvh_histogram_percent_quantiles_by_dose_bin,
                                       "Quantiles volume dict": differential_dvh_histogram_volume_quantiles_by_dose_bin} # note that all rows in the edges array should be equal!

                # compute cumulative dvh quantities from differential dvh
                cumulative_dvh_counts_by_MC_trial_arr_D0_val = np.sum(differential_dvh_histogram_counts_by_MC_trial_arr, axis=1, keepdims = True)
                cumulative_dvh_counts_by_MC_trial_arr_Dgt0_vals = num_sampled_bx_pts - np.cumsum(differential_dvh_histogram_counts_by_MC_trial_arr, axis=1)
                cumulative_dvh_counts_by_MC_trial_arr = np.concatenate((cumulative_dvh_counts_by_MC_trial_arr_D0_val, cumulative_dvh_counts_by_MC_trial_arr_Dgt0_vals), axis=1)
                cumulative_dvh_volume_by_MC_trial_arr = cumulative_dvh_counts_by_MC_trial_arr*bx_sample_pts_volume_element
                cumulative_dvh_percent_by_MC_trial_arr = (cumulative_dvh_counts_by_MC_trial_arr/num_sampled_bx_pts)*100
                
                cumulative_dvh_dose_vals_by_MC_trial_1darr = differential_dvh_histogram_edges_by_MC_trial_arr[0].copy()

                # Note that the nominal is excluded from the calculation
                cumulative_dvh_histogram_counts_quantiles_by_dose_val = {'Q'+str(q): np.quantile(cumulative_dvh_counts_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                cumulative_dvh_histogram_volume_quantiles_by_dose_val = {'Q'+str(q): np.quantile(cumulative_dvh_volume_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                cumulative_dvh_histogram_percent_quantiles_by_dose_val = {'Q'+str(q): np.quantile(cumulative_dvh_percent_by_MC_trial_arr[1:], q/100,axis=0) for q in range(5,100,5)}
                
                # Note that the 0th index is the nominal value for the counts, percent and volumes arrays
                cumulative_dvh_dict = {"Counts arr": cumulative_dvh_counts_by_MC_trial_arr, 
                                       "Percent arr": cumulative_dvh_percent_by_MC_trial_arr, 
                                       "Volume arr (cubic mm)": cumulative_dvh_volume_by_MC_trial_arr, 
                                       "Dose vals arr (Gy)": cumulative_dvh_dose_vals_by_MC_trial_1darr,
                                       "Quantiles counts dict": cumulative_dvh_histogram_counts_quantiles_by_dose_val,
                                       "Quantiles percent dict": cumulative_dvh_histogram_percent_quantiles_by_dose_val,
                                       "Quantiles volume dict": cumulative_dvh_histogram_volume_quantiles_by_dose_val}
                
                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###

                specific_bx_structure["MC data: Differential DVH dict"] = differential_dvh_dict
                specific_bx_structure["MC data: Cumulative DVH dict"] = cumulative_dvh_dict 
                specific_bx_structure["MC data: dose volume metrics dict"] = dvh_metric_vol_dose_percent_dict 

                biopsies_progress.update(calculate_biopsy_DVH_quantities_by_biopsy_task, advance = 1)
            biopsies_progress.remove_task(calculate_biopsy_DVH_quantities_by_biopsy_task)    
            patients_progress.update(calculate_biopsy_DVH_quantities_task, advance=1)
            completed_progress.update(calculate_biopsy_DVH_quantities_task_complete, advance=1)
        patients_progress.update(calculate_biopsy_DVH_quantities_task, visible = False)
        completed_progress.update(calculate_biopsy_DVH_quantities_task_complete, visible = True)
        live_display.refresh()

        """

        #### MAY NEED TO BRING THIS SECTION BACK, BUT I THINK EVERYTHING CAN BE MIGRATED NOW TO NEWER FUNCTIONS IF AN ERROR IS ENCOUNTERED DOWN THE PIPELINE
        """
        computing_MLE_statistics_dose_task = patients_progress.add_task("[red]Computing dosimetric localization statistics (MLE) [{}]...".format("initializing"), total=num_patients)
        computing_MLE_statistics_dose_task_complete = completed_progress.add_task("[green]Computing dosimetric localization statistics (MLE)", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(computing_MLE_statistics_dose_task, description = "[red]Computing dosimetric localization statistics (MLE) [{}]...".format(patientUID))
            
            if dose_ref not in pydicom_item:
                patients_progress.update(computing_MLE_statistics_dose_task, advance = 1)
                completed_progress.update(computing_MLE_statistics_dose_task_complete, advance = 1)
                continue
            
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            compile_results_dose_NN_biopsy_containment_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                dosimetric_localization_dose_vals_by_bx_point_all_trials_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"][:,1:] # REMOVE NOMINAL!
                dosimetric_localization_dose_vals_by_bx_point_all_trials_list = dosimetric_localization_dose_vals_by_bx_point_all_trials_arr.tolist()

                dosimetric_MLE_statistics_all_bx_pts_list = normal_distribution_MLE_parallel(parallel_pool, dosimetric_localization_dose_vals_by_bx_point_all_trials_list)
                mu_se_var_all_bx_pts_list = [bx_point_stats[0] for bx_point_stats in dosimetric_MLE_statistics_all_bx_pts_list]
                confidence_intervals_all_bx_pts_list = [bx_point_stats[1] for bx_point_stats in dosimetric_MLE_statistics_all_bx_pts_list]
                
                mu_all_bx_pts_list = np.mean(dosimetric_localization_dose_vals_by_bx_point_all_trials_list, axis = 1).tolist()
                std_all_bx_pts_list = np.std(dosimetric_localization_dose_vals_by_bx_point_all_trials_list, axis = 1, ddof=1).tolist()
                quantiles_all_bx_pts_dict_of_lists = {'Q'+str(q): np.quantile(dosimetric_localization_dose_vals_by_bx_point_all_trials_list, q/100,axis = 1).tolist() for q in range(5,100,5)}

                MC_dose_stats_dict = {"Dose statistics by bx pt (mean,se,var)": mu_se_var_all_bx_pts_list, "Confidence intervals (95%) by bx pt": confidence_intervals_all_bx_pts_list}
                MC_dose_stats_basic_dict = {"Mean dose by bx pt": mu_all_bx_pts_list, "STD by bx pt": std_all_bx_pts_list, "Quantiles dose by bx pt dict": quantiles_all_bx_pts_dict_of_lists}
                specific_bx_structure["MC data: Dose statistics (MLE) for each sampled bx pt list (mean, std)"] = MC_dose_stats_dict
                specific_bx_structure["MC data: Dose statistics for each sampled bx pt list (mean, std, quantiles)"] = MC_dose_stats_basic_dict
                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, advance = 1)
            
            biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, visible = False)
            patients_progress.update(computing_MLE_statistics_dose_task, advance = 1)
            completed_progress.update(computing_MLE_statistics_dose_task_complete, advance = 1)

        patients_progress.update(computing_MLE_statistics_dose_task, visible = False)
        completed_progress.update(computing_MLE_statistics_dose_task_complete, visible = True)
        """
        #### MAY NEED TO BRING THIS SECTION BACK

        
        """

        # voxelize dose results
        biopsy_voxelize_dose_task = patients_progress.add_task("[red]Voxelizing dose results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_dose_task_complete = completed_progress.add_task("[green]Voxelizing dose results", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing dose results [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_dose_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_dose_results_arr = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"][:,1:]
                specific_bx_dose_results_list = specific_bx_dose_results_arr.tolist()
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
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
                    
                    total_num_MC_trials_in_voxel = len(dose_vals_in_voxel_list)
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


        """

        """

        ### VOXELIZE DOSE RESULTS NEW WAY!

        # voxelize dose results
        biopsy_voxelize_dose_task = patients_progress.add_task("[red]Voxelizing dose results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_dose_task_complete = completed_progress.add_task("[green]Voxelizing dose results", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing dose results [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_dose_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                
                ### Prelims

                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Prelims", total = None)
                ###

                specific_bx_dose_results_arr = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"][:,1:]
                specific_bx_dose_results_list = specific_bx_dose_results_arr.tolist()
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                randomly_sampled_bx_pts_bx_coord_sys_arr = specific_bx_structure['Random uniformly sampled volume pts bx coord sys arr']
                biopsy_cyl_z_length = specific_bx_structure["Reconstructed biopsy cylinder length (from contour data)"]
                num_z_voxels = float(math.floor(float(biopsy_cyl_z_length/biopsy_z_voxel_length)))
                constant_voxel_biopsy_cyl_z_length = num_z_voxels*biopsy_z_voxel_length
                biopsy_z_length_difference = biopsy_cyl_z_length - constant_voxel_biopsy_cyl_z_length
                extra_length_for_biopsy_end_cap_voxels = biopsy_z_length_difference/2

                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing", total = None)
                ###
                
                voxel_z_begin = 0.
                voxelized_biopsy_dose_results_list = [None]*int(num_z_voxels)
                voxel_dose_dict_empty = {"Voxel z begin": None, 
                                         "Voxel z end": None, 
                                         "Indices from all sample pts that are within voxel arr": None, 
                                         "Num sample pts in voxel": None, 
                                         "Sample pts in voxel arr (bx coord sys)": None, 
                                         "All dose vals in voxel list": None, 
                                         "Total num MC trials in voxel": None, 
                                         "Arithmetic mean of dose in voxel": None, 
                                         "Std dev of dose in voxel": None
                                         }
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
                    
                    total_num_MC_trials_in_voxel = len(dose_vals_in_voxel_list)
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


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~ Reorganizing", total = None)
                ###
                
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


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###
                
                specific_bx_structure["MC data: voxelized dose results list"] = voxelized_biopsy_dose_results_list
                specific_bx_structure["MC data: voxelized dose results dict (dict of lists)"] = voxelized_biopsy_dose_results_dict
                 
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, advance = 1)
            biopsies_progress.remove_task(biopsy_voxelize_each_bx_structure_dose_task)
            patients_progress.update(biopsy_voxelize_dose_task, advance = 1)
            completed_progress.update(biopsy_voxelize_dose_task_complete, advance = 1)
        patients_progress.update(biopsy_voxelize_dose_task, visible = False)
        completed_progress.update(biopsy_voxelize_dose_task_complete,visible = True)
        live_display.refresh()



        """









        """
  
        # voxelize dose results
        biopsy_voxelize_dose_task = patients_progress.add_task("[red]Voxelizing dose results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_dose_task_complete = completed_progress.add_task("[green]Voxelizing dose results", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_dose_task, description = "[red]Voxelizing dose results [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID][all_ref_key]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_dose_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                
                ### Prelims

                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Prelims", total = None)
                ###

                specific_bx_dose_results_arr = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: Dose vals for each sampled bx pt arr (nominal & all MC trials)"][:,1:]
                specific_bx_dose_results_list = specific_bx_dose_results_arr.tolist()
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                
                randomly_sampled_bx_pts_bx_coord_sys_arr = specific_bx_structure['Random uniformly sampled volume pts bx coord sys arr']
                biopsy_cyl_z_length = specific_bx_structure["Reconstructed biopsy cylinder length (from contour data)"]
                num_z_voxels = float(math.floor(float(biopsy_cyl_z_length/biopsy_z_voxel_length)))
                constant_voxel_biopsy_cyl_z_length = num_z_voxels*biopsy_z_voxel_length
                biopsy_z_length_difference = biopsy_cyl_z_length - constant_voxel_biopsy_cyl_z_length
                extra_length_for_biopsy_end_cap_voxels = biopsy_z_length_difference/2

                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Calcing", total = None)
                ###
                
                voxel_z_begin = 0.
                voxelized_biopsy_dose_results_list = [None]*int(num_z_voxels)
                voxel_dose_dict_empty = {"Voxel z begin": None, 
                                         "Voxel z end": None, 
                                         "Indices from all sample pts that are within voxel arr": None, 
                                         "Num sample pts in voxel": None, 
                                         "Sample pts in voxel arr (bx coord sys)": None, 
                                         "All dose vals in voxel list": None, 
                                         "Total num MC trials in voxel": None, 
                                         "Arithmetic mean of dose in voxel": None, 
                                         "Std dev of dose in voxel": None
                                         }
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
                    
                    total_num_MC_trials_in_voxel = len(dose_vals_in_voxel_list)
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


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###



                ###
                indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~ Reorganizing", total = None)
                ###
                
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


                ###
                indeterminate_progress_sub.update(indeterminate_task, visible = False)
                ###
                
                specific_bx_structure["MC data: voxelized dose results list"] = voxelized_biopsy_dose_results_list
                specific_bx_structure["MC data: voxelized dose results dict (dict of lists)"] = voxelized_biopsy_dose_results_dict
                 
                biopsies_progress.update(biopsy_voxelize_each_bx_structure_dose_task, advance = 1)
            biopsies_progress.remove_task(biopsy_voxelize_each_bx_structure_dose_task)
            patients_progress.update(biopsy_voxelize_dose_task, advance = 1)
            completed_progress.update(biopsy_voxelize_dose_task_complete, advance = 1)
        patients_progress.update(biopsy_voxelize_dose_task, visible = False)
        completed_progress.update(biopsy_voxelize_dose_task_complete,visible = True)
        live_display.refresh()
        """

        
        master_structure_info_dict['Global']["MC info"]['MC containment sim performed'] = perform_mc_containment_sim
        master_structure_info_dict['Global']["MC info"]['MC dose sim performed'] = perform_mc_dose_sim
        
        

        return master_structure_reference_dict, master_structure_info_dict, live_display
    

















################ BEGIN FUNCTIONS














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
    



def dosimetric_localization_parallel(parallel_pool, 
                                     bx_only_shifted_3darr, 
                                     specific_bx_structure, 
                                     dose_ref_dict, 
                                     dose_ref, 
                                     phys_space_dose_map_phys_coords_2d_arr, 
                                     phys_space_dose_map_dose_2d_arr, 
                                     num_dose_calc_NN):
    # build args list
    num_trials = bx_only_shifted_3darr.shape[0]
    args_list = [None]*(num_trials) 
    dose_ref_dict_roi = dose_ref_dict["Dose ID"]
    specific_bx_structure_roi = specific_bx_structure["ROI"]
    dose_data_KDtree = dose_ref_dict["KDtree"]
    for single_MC_trial_slice_index, bx_only_shifted_single_MC_trial_slice in enumerate(bx_only_shifted_3darr):
        """
        if single_MC_trial_slice_index > (num_MC_dose_simulations-1):
            break
        else:
            pass 
        """
        single_MC_trial_arg = (dose_data_KDtree, bx_only_shifted_single_MC_trial_slice, specific_bx_structure_roi, dose_ref_dict_roi, dose_ref, phys_space_dose_map_phys_coords_2d_arr, phys_space_dose_map_dose_2d_arr, num_dose_calc_NN)
        args_list[single_MC_trial_slice_index] = single_MC_trial_arg
    
    # conduct the dosimetric localization in parallel. The MC trials are done in parallel.
    dosimetric_localiation_all_MC_trials_list = parallel_pool.starmap(dosimetric_localization_single_MC_trial, args_list)

    return dosimetric_localiation_all_MC_trials_list


def dosimetric_localization_non_parallel(bx_only_shifted_3darr, 
                                         specific_bx_structure, 
                                         dose_ref_dict, 
                                         dose_ref, 
                                         phys_space_dose_map_phys_coords_2d_arr, 
                                         phys_space_dose_map_dose_2d_arr, 
                                         num_dose_calc_NN):
    # build args list
    num_trials = bx_only_shifted_3darr.shape[0]
    args_list = [None]*(num_trials) 
    dose_ref_dict_roi = dose_ref_dict["Dose ID"]
    specific_bx_structure_roi = specific_bx_structure["ROI"]
    dose_data_KDtree = dose_ref_dict["KDtree"]
    for single_MC_trial_slice_index, bx_only_shifted_single_MC_trial_slice in enumerate(bx_only_shifted_3darr):
        """
        if single_MC_trial_slice_index > (num_MC_dose_simulations-1):
            break
        else:
            pass 
        """
        single_MC_trial_arg = (dose_data_KDtree, bx_only_shifted_single_MC_trial_slice, specific_bx_structure_roi, dose_ref_dict_roi, dose_ref, phys_space_dose_map_phys_coords_2d_arr, phys_space_dose_map_dose_2d_arr, num_dose_calc_NN)
        args_list[single_MC_trial_slice_index] = single_MC_trial_arg
    
    # conduct the dosimetric localization in parallel. The MC trials are done in parallel.
    dosimetric_localiation_all_MC_trials_list = [None]*len(args_list)
    for index, arg in enumerate(args_list):
        nearest_neighbours_single_MC_trial_NN_parent_obj = dosimetric_localization_single_MC_trial(*arg)
        dosimetric_localiation_all_MC_trials_list[index] = nearest_neighbours_single_MC_trial_NN_parent_obj

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


def MC_simulator_shift_biopsy_structures_uniform_generator_parallel(parallel_pool, patient_dict, structs_referenced_list, bx_ref, biopsy_needle_compartment_length, num_simulations):
    # build args list for parallel computing
    args_list = []
    bx_structs = bx_ref
    for specific_bx_structure_index, specific_bx_structure in enumerate(patient_dict[bx_structs]):
        #spec_structure_zslice_wise_delaunay_obj_list = specific_structure["Delaunay triangulation zslice-wise list"]
        bx_core_length = specific_bx_structure['Reconstructed biopsy cylinder length (from contour data)']
        core_length_compartment_length_difference = biopsy_needle_compartment_length - bx_core_length
        if core_length_compartment_length_difference <= 0:
            core_length_compartment_length_difference = 0.
        
        specific_bx_structure_args = (bx_structs, specific_bx_structure_index, core_length_compartment_length_difference, num_simulations)
        args_list.append(specific_bx_structure_args)

    sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list = parallel_pool.starmap(MC_simulator_uniform_shift_structure_generator,args_list)

    # update the patient dictionary
    for generated_shifts_info_list in sp_bx_structure_uniform_dist_shift_samples_and_structure_reference_list:
        structure_type = generated_shifts_info_list[0]
        specific_structure_index = generated_shifts_info_list[1]
        specific_structure_structure_uniform_dist_shift_samples_arr = generated_shifts_info_list[2]
        patient_dict[structure_type][specific_structure_index]["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"] = specific_structure_structure_uniform_dist_shift_samples_arr



def MC_simulator_shift_all_structures_generator_parallel(parallel_pool, patient_dict, structs_referenced_list, num_simulations):

    # build args list for parallel computing
    args_list = []
    #patient_dict_updated_with_generated_samples = patient_dict.copy()
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
        patient_dict[structure_type][specific_structure_index]["MC data: Generated normal dist random samples arr"] = specific_structure_structure_normal_dist_shift_samples_arr



def MC_simulator_shift_structure_generator(structure_type, specific_structure_index, sp_struct_uncertainty_data_mean_arr, sp_struct_uncertainty_data_sigma_arr, num_simulations):
    structure_normal_dist_shift_samples_arr = np.array([ 
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[0], scale=sp_struct_uncertainty_data_sigma_arr[0], size=num_simulations),  
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[1], scale=sp_struct_uncertainty_data_sigma_arr[1], size=num_simulations),  
            np.random.normal(loc=sp_struct_uncertainty_data_mean_arr[2], scale=sp_struct_uncertainty_data_sigma_arr[2], size=num_simulations)],   
            dtype = float).T
    generated_shifts_info_list = [structure_type, specific_structure_index, structure_normal_dist_shift_samples_arr]
    return generated_shifts_info_list



def MC_simulator_uniform_shift_structure_generator(structure_type, specific_structure_index, core_length_compartment_length_difference, num_simulations):
    """
    note that since this is an uncertainty in the location of the biopsy within the compartment,
    shifts in the x and y directions (relative to the biopsy reconstructed core) are zero, while
    shifts in the z direction (relative to the biopsy reconstructed core, ie along the axis of the needle) 
    are uniformly generated. Therefore, we just generate distances here, and that will be multipled by
    the unit vector along the needle axis later.
    """
    """
    #this code was used to generate 3d vectors, but there is no need, as described in the function description
    structure_uniform_dist_shift_samples_arr = np.array([ 
            np.zeros(num_simulations, dtype=float),  
            np.zeros(num_simulations, dtype=float),  
            np.random.uniform(low=0, high=core_length_compartment_length_difference, size=num_simulations)],   
            dtype = float).T
    """
    structure_uniform_dist_shift_samples_arr = np.random.uniform(low=0, high=core_length_compartment_length_difference, size=num_simulations)

    generated_shifts_info_list = [structure_type, specific_structure_index, structure_uniform_dist_shift_samples_arr]
    return generated_shifts_info_list


def MC_simulator_translate_sampled_bx_points_arr_bx_only_shift_parallel(parallel_pool, specific_bx_structure, simulate_uniform_bx_shifts_due_to_bx_needle_compartment, plot_uniform_shifts_to_check_plotly):
    randomly_sampled_bx_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
    randomly_sampled_bx_shifts_arr = specific_bx_structure["MC data: Generated normal dist random samples arr"]
    
    # do the normal only shifts alone, regardless of whether uniform shifts will be done, data may be useful later
    args_list_normal_shifts = []
    for bx_shift in randomly_sampled_bx_shifts_arr:
        arg = (randomly_sampled_bx_pts_arr, bx_shift)
        args_list_normal_shifts.append(arg)
    
    randomly_sampled_bx_pts_arr_bx_normal_only_shift_arr_each_sampled_shift_list = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_bx_only_shift,args_list_normal_shifts)
    
    if simulate_uniform_bx_shifts_due_to_bx_needle_compartment == True:
        random_uniformly_sampled_bx_shifts_arr = specific_bx_structure["MC data: Generated uniform dist (biopsy needle compartment) random distance (z_needle) samples arr"]
        # notice the minus sign below!!
        bx_needle_centroid_vec_tip_to_handle_unit_vec = -specific_bx_structure["Centroid line unit vec (bx needle base to bx needle tip)"]
        num_uniform_shifts = random_uniformly_sampled_bx_shifts_arr.shape[0]
        bx_needle_uniform_compartment_shift_vectors_array = np.tile(bx_needle_centroid_vec_tip_to_handle_unit_vec,(num_uniform_shifts,1))
        bx_needle_uniform_compartment_shift_vectors_array = np.multiply(bx_needle_uniform_compartment_shift_vectors_array,random_uniformly_sampled_bx_shifts_arr[...,None]) # The [...,None] converts the row vector to a column vector for proper element-wise multiplication
        specific_bx_structure["MC data: Generated uniform (biopsy needle compartment) random vectors (z_needle) samples arr"] = bx_needle_uniform_compartment_shift_vectors_array
        total_rigid_shift_vectors_arr = bx_needle_uniform_compartment_shift_vectors_array + randomly_sampled_bx_shifts_arr
        specific_bx_structure["MC data: Total rigid shift vectors arr"] = total_rigid_shift_vectors_arr
        args_list_uniform_compartment_shifts = []
        for uniform_bx_shift_distance in random_uniformly_sampled_bx_shifts_arr:
            arg = (randomly_sampled_bx_pts_arr, uniform_bx_shift_distance, bx_needle_centroid_vec_tip_to_handle_unit_vec, plot_uniform_shifts_to_check_plotly)
            args_list_uniform_compartment_shifts.append(arg)
        
        randomly_sampled_bx_pts_bx_uniform_compartment_only_shift_arr_each_sampled_shift_list = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_bx_uniform_compartment_only_shift, args_list_uniform_compartment_shifts)
 
        args_list_uniform_and_normal_shifts = []
        for bx_shift_ind, bx_shift in enumerate(randomly_sampled_bx_shifts_arr):
            arg = (randomly_sampled_bx_pts_bx_uniform_compartment_only_shift_arr_each_sampled_shift_list[bx_shift_ind], bx_shift)
            args_list_uniform_and_normal_shifts.append(arg)
    
        randomly_sampled_bx_pts_arr_bx_only_shift_final_arr_each_sampled_shift_list = parallel_pool.starmap(MC_simulator_translate_sampled_bx_points_arr_bx_only_shift,args_list_uniform_and_normal_shifts)
    else:
        total_rigid_shift_vectors_arr = randomly_sampled_bx_shifts_arr
        specific_bx_structure["MC data: Total rigid shift vectors arr"] = total_rigid_shift_vectors_arr
        randomly_sampled_bx_pts_arr_bx_only_shift_final_arr_each_sampled_shift_list = randomly_sampled_bx_pts_arr_bx_normal_only_shift_arr_each_sampled_shift_list.copy()
        
    num_bx_sampled_points = randomly_sampled_bx_pts_arr.shape[0]
    num_bx_sampled_shifts = randomly_sampled_bx_shifts_arr.shape[0]

    """
    create a 3d array that stores all the shifted bx data where each 3d slice is the shifted bx data for 
    a fixed sampled bx shift, ie each slice is a sampled bx shift trial. Note though that if the uniform 
    compartment shifts are done, these are still slices of constant shift vectors, but they are now 
    composed of two shifts. Each slice is a unique uniform shift vector plus a unique norm shift vector.
    """
    randomly_sampled_bx_pts_arr_bx_only_shift_3Darr = np.empty([num_bx_sampled_shifts,num_bx_sampled_points,3],dtype=float)

    # build the above described 3d array from the parallel results list
    for index, randomly_sampled_bx_pts_arr_bx_only_shift_arr in enumerate(randomly_sampled_bx_pts_arr_bx_only_shift_final_arr_each_sampled_shift_list):
        randomly_sampled_bx_pts_arr_bx_only_shift_3Darr[index] = randomly_sampled_bx_pts_arr_bx_only_shift_arr

    return randomly_sampled_bx_pts_arr_bx_only_shift_3Darr


def MC_simulator_translate_sampled_bx_points_arr_bx_uniform_compartment_only_shift(randomly_sampled_bx_pts_arr, bx_uniform_shift_length, bx_centroid_vec_tip_to_needle_handle_unit_vec, plot_uniform_shifts_to_check = True):
    # For a single trial, all BX points are shifted by the same vector!
    randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_arr = randomly_sampled_bx_pts_arr + bx_centroid_vec_tip_to_needle_handle_unit_vec*bx_uniform_shift_length
    """
    these should shift inferior (more negative values) if the biopsies are 
    systematically being contoured at the biopsy needle tip.
    """
    if plot_uniform_shifts_to_check == True:
        plotting_funcs.plotly_3dscatter_arbitrary_number_of_arrays([randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_arr, randomly_sampled_bx_pts_arr], colors_for_arrays_list = ['blue','red'], aspect_mode_input = 'data')
    
    return randomly_sampled_bx_pts_arr_bx_uniform_compartment_only_shift_arr


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



def point_containment_test_axis_aligned_bounding_box_and_zslice_wise_2d_PIP_parallel(parallel_pool, containment_structure_pts_arr, interslice_interpolation_information_of_containment_structure, test_pts_arr):
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


def random_uniform_point_sampler_from_global_delaunay_convex_structure_parallel(num_samples, delaunay_global_convex_structure_tri, reconstructed_bx_arr, patientUID, structure_type, specific_structure_index):
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


def grid_point_sampler_from_global_delaunay_convex_structure_parallel(grid_separation_distance, delaunay_global_convex_structure_tri, reconstructed_bx_arr, patientUID, structure_type, specific_structure_index):
    
    reconstructed_bx_point_cloud = point_containment_tools.create_point_cloud(reconstructed_bx_arr)
    reconstructed_bx_point_cloud_color = np.array([0,0,1])
    reconstructed_bx_point_cloud.paint_uniform_color(reconstructed_bx_point_cloud_color)

    
    axis_aligned_bounding_box = reconstructed_bx_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    lattice_sizex = int(math.ceil(abs(max_bounds[0]-min_bounds[0])/grid_separation_distance) + 1)
    lattice_sizey = int(math.ceil(abs(max_bounds[1]-min_bounds[1])/grid_separation_distance) + 1)
    lattice_sizez = int(math.ceil(abs(max_bounds[2]-min_bounds[2])/grid_separation_distance) + 1)
    origin = min_bounds

    bx_samples_arr = generate_cubic_lattice(grid_separation_distance, lattice_sizex,lattice_sizey,lattice_sizez,origin)
    list_of_passed_indices = []
    for test_point_index, test_point in enumerate(bx_samples_arr):
        containment_result_bool = point_containment_tools.convex_bx_structure_global_test_point_containment(delaunay_global_convex_structure_tri,test_point)
        if containment_result_bool == True:
            list_of_passed_indices.append(test_point_index)
        else:
            pass
    num_pts_contained = len(list_of_passed_indices)
    bx_samples_arr_within_bx = np.empty((num_pts_contained,3),dtype=float)

    for index_for_new_arr,index_in_org_array in enumerate(list_of_passed_indices):
        bx_samples_arr_within_bx[index_for_new_arr] = bx_samples_arr[index_in_org_array]

    num_bx_sample_pts_after_exclusions = bx_samples_arr_within_bx.shape[0]

    bx_samples_arr_point_cloud_color = np.random.uniform(0, 0.7, size=3)
    bx_samples_arr_point_cloud = point_containment_tools.create_point_cloud(bx_samples_arr_within_bx, bx_samples_arr_point_cloud_color)
    
    return bx_samples_arr_within_bx, axis_aligned_bounding_box_points_arr, num_bx_sample_pts_after_exclusions, {"Patient UID": patientUID, "Structure type": structure_type, "Specific structure index": specific_structure_index}
    

def grid_point_sampler_rotated_from_global_delaunay_convex_structure_parallel(grid_separation_distance, 
                                                                              delaunay_global_convex_structure_tri, 
                                                                              reconstructed_bx_arr, 
                                                                              patientUID, 
                                                                              structure_type, 
                                                                              specific_structure_index,
                                                                              z_axis_to_centroid_vec_rotation_matrix
                                                                              ):
    
    reconstructed_bx_point_cloud = point_containment_tools.create_point_cloud(reconstructed_bx_arr)
    reconstructed_bx_point_cloud_color = np.array([0,0,1])
    reconstructed_bx_point_cloud.paint_uniform_color(reconstructed_bx_point_cloud_color)

    
    axis_aligned_bounding_box = reconstructed_bx_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    lattice_sizex = int(math.ceil(abs(max_bounds[0]-min_bounds[0])/grid_separation_distance) + 1)
    lattice_sizey = int(math.ceil(abs(max_bounds[1]-min_bounds[1])/grid_separation_distance) + 1)
    lattice_sizez = int(math.ceil(abs(max_bounds[2]-min_bounds[2])/grid_separation_distance) + 1)
    origin = min_bounds

    bx_samples_arr_unrotated = generate_cubic_lattice(grid_separation_distance, lattice_sizex,lattice_sizey,lattice_sizez,origin)
    bx_samples_arr_rotated = (z_axis_to_centroid_vec_rotation_matrix @ bx_samples_arr_unrotated.T).T
    grid_sample_rotated_pcd = point_containment_tools.create_point_cloud(bx_samples_arr_rotated)
    grid_sample_rotated_pcd.paint_uniform_color(np.array([1,0,0]))
    reconstructed_bx_point_cloud_global_center_arr = reconstructed_bx_point_cloud.get_center()
    grid_sample_point_cloud_global_center_arr = grid_sample_rotated_pcd.get_center()
    grid_sample_to_biopsy_translation_vec = reconstructed_bx_point_cloud_global_center_arr - grid_sample_point_cloud_global_center_arr
    bx_samples_arr = bx_samples_arr_rotated + grid_sample_to_biopsy_translation_vec


    grid_sample_final_pcd = point_containment_tools.create_point_cloud(bx_samples_arr)
    grid_sample_final_pcd.paint_uniform_color(np.array([1,0,1]))

    grid_sample_nonrotated_pcd = point_containment_tools.create_point_cloud(bx_samples_arr_unrotated)
    grid_sample_nonrotated_pcd.paint_uniform_color(np.array([0,1,0]))
    
    
    #plotting_funcs.plot_geometries(grid_sample_rotated_pcd,grid_sample_final_pcd,reconstructed_bx_point_cloud)


    list_of_passed_indices = []
    for test_point_index, test_point in enumerate(bx_samples_arr):
        containment_result_bool = point_containment_tools.convex_bx_structure_global_test_point_containment(delaunay_global_convex_structure_tri,test_point)
        if containment_result_bool == True:
            list_of_passed_indices.append(test_point_index)
        else:
            pass
    num_pts_contained = len(list_of_passed_indices)
    bx_samples_arr_within_bx = np.empty((num_pts_contained,3),dtype=float)

    for index_for_new_arr,index_in_org_array in enumerate(list_of_passed_indices):
        bx_samples_arr_within_bx[index_for_new_arr] = bx_samples_arr[index_in_org_array]

    num_bx_sample_pts_after_exclusions = bx_samples_arr_within_bx.shape[0]

    bx_samples_arr_point_cloud_color = np.random.uniform(0, 0.7, size=3)
    bx_samples_arr_point_cloud = point_containment_tools.create_point_cloud(bx_samples_arr_within_bx, bx_samples_arr_point_cloud_color)
    
    return bx_samples_arr_within_bx, axis_aligned_bounding_box_points_arr, num_bx_sample_pts_after_exclusions, {"Patient UID": patientUID, "Structure type": structure_type, "Specific structure index": specific_structure_index}


def generate_cubic_lattice(distance, sizex,sizey,sizez,origin):
    """
    Generates an evenly spaced cubic lattice of points in three dimensions.

    Returns:
        numpy.ndarray: Array of lattice points.
    """
    total_points = int(sizex*sizey*sizez)
    points = np.zeros((total_points, 3))
    
    index = 0
    for i in range(sizex):
        for j in range(sizey):
            for k in range(sizez):
                points[index] = np.array([i, j, k]) * distance
                index += 1
    points = points + origin
    return points

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
        comparison_structure_NN_distances_reciprocal = np.reciprocal(comparison_structure_NN_distances)
        comparison_structure_NN_indices = self.NN_search_output[1]
        nearest_points_on_comparison_struct = comparison_structure_points_that_made_KDtree[comparison_structure_NN_indices]
        nearest_doses_list = self.dose_arr[comparison_structure_NN_indices]
        nearest_doses_weighted_mean_list = np.average(nearest_doses_list, axis=1,weights = comparison_structure_NN_distances_reciprocal).tolist()
        NN_data_list = [nearest_neighbour_child_dose(self.queried_Bx_points[index], nearest_points_on_comparison_struct[index], nearest_doses_list[index], comparison_structure_NN_distances[index], nearest_doses_weighted_mean_list[index]) for index in range(0,len(self.queried_Bx_points))]
        #NN_data_list = [{"Queried BX pt": self.queried_Bx_points[index], "NN pt on comparison struct": nearest_points_on_comparison_struct[index], "Euclidean distance": comparison_structure_NN_distances[index]} for index in range(0,len(self.queried_Bx_points))]
        return NN_data_list


class nearest_neighbour_child_dose:
    def __init__(self, queried_BX_pt, NN_pt_on_comparison_struct, NN_dose_on_comparison_struct, euclidean_dist, nearest_dose):
        self.queried_BX_pt = queried_BX_pt
        self.NN_pt_on_comparison_struct = NN_pt_on_comparison_struct
        self.NN_dose_on_comparison_struct = NN_dose_on_comparison_struct
        self.euclidean_dist = euclidean_dist
        self.nearest_dose = nearest_dose


def create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list):
    structure_organized_for_bx_data_blank_dict = {}
    for non_bx_struct_type in structs_referenced_list[1:]:
        for specific_non_bx_structure_index, specific_non_bx_structure in enumerate(pydicom_item[non_bx_struct_type]):
            specific_non_bx_struct_roi = specific_non_bx_structure["ROI"]
            specific_non_bx_struct_refnum = specific_non_bx_structure["Ref #"]
            structure_organized_for_bx_data_blank_dict[specific_non_bx_struct_roi,non_bx_struct_type,specific_non_bx_struct_refnum,specific_non_bx_structure_index] = None

    return structure_organized_for_bx_data_blank_dict


def tissue_length_calculator(z_coords_arr,
                             binom_est_arr,
                             binom_est_se_arr,
                             lattice_spacing, 
                             threshold_probability = 0.9,
                             bootstraps = 100):
    
    threshold = threshold_probability
    n = bootstraps
    z_coords = z_coords_arr
    binom_ests = binom_est_arr
    binom_se = binom_est_se_arr

    data = np.array([z_coords,binom_ests,binom_se])
    sorted_data = data[:,data[0,:].argsort(axis=0)] 

    sample_matrix = np.empty([n,sorted_data.shape[1]])

    for i in range(sample_matrix.shape[0]):
        for j in range(sample_matrix.shape[1]):
            sample_matrix[i,j] = np.random.normal(loc = sorted_data[1,j], scale = sorted_data[2,j])

    length_estimate_distribution = np.empty(sample_matrix.shape[0])
    for i in range(sample_matrix.shape[0]):
        particular_sample_1darr = sample_matrix[i]
        delta_z_tot = 0
        z_start_found = False
        for index,binom_est in enumerate(particular_sample_1darr):
            if binom_est >= threshold:
                if z_start_found == True:
                    pass
                else: 
                    z_start = sorted_data[0][index]
                    z_start_found = True
            else: 
                if z_start_found == True:
                    z_end = sorted_data[0][index-1]
                    z_diff = z_end - z_start
                    if z_diff > lattice_spacing:
                        delta_z_tot = delta_z_tot + z_diff
                    else:
                        delta_z_tot = delta_z_tot + lattice_spacing
                    z_start_found = False
                else:
                    pass
        
        length_estimate_distribution[i] = delta_z_tot
    
    # calculate statistics
    length_estimate_mean = np.mean(length_estimate_distribution)
    length_estimate_std = np.std(length_estimate_distribution)
    length_estimate_se = length_estimate_std/np.sqrt(length_estimate_distribution.shape[0])

    return length_estimate_distribution, length_estimate_mean, length_estimate_se



def tissue_volume_calculator(patientUID,
                             specific_bx_structure,
                             cancer_tissue_label,
                             probability_threshold,
                             bx_sample_pts_volume_element,
                             structure_miss_probability_roi):
    
    bx_roi = specific_bx_structure["ROI"]
    bx_sim_bool = specific_bx_structure["Simulated bool"]
    bx_type = specific_bx_structure["Simulated type"]
    containment_output_by_MC_trial_pandas_data_frame = specific_bx_structure["Output data frames"]["Mutual containment output by bx point"]  
    total_structure_volume = specific_bx_structure["Structure volume"]

    # volume of dil tissue
    # need to convert this particular column back to float so that we can compare >=
    #containment_output_by_MC_trial_pandas_data_frame = misc_tools.convert_categorical_columns(containment_output_by_MC_trial_pandas_data_frame, ["Mean probability (binom est)"], [float])
    containment_output_by_MC_trial_pandas_data_frame_DIL_and_min_threshold_subset = containment_output_by_MC_trial_pandas_data_frame[(containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == cancer_tissue_label) &
                                                    (containment_output_by_MC_trial_pandas_data_frame["Mean probability (binom est)"] >= probability_threshold)]    
    num_dil_voxels_in_subset = containment_output_by_MC_trial_pandas_data_frame_DIL_and_min_threshold_subset.shape[0]
    volume_of_dil_tissue = num_dil_voxels_in_subset*bx_sample_pts_volume_element

    # volume of miss structure tissue
    containment_output_by_MC_trial_pandas_data_frame_miss_structure_and_min_threshold_subset = containment_output_by_MC_trial_pandas_data_frame[(containment_output_by_MC_trial_pandas_data_frame["Structure ROI"] == structure_miss_probability_roi) &
                                                    (containment_output_by_MC_trial_pandas_data_frame["Mean probability (binom est)"] >= probability_threshold)]    
    num_miss_structure_voxels_in_subset = containment_output_by_MC_trial_pandas_data_frame_miss_structure_and_min_threshold_subset.shape[0]
    volume_of_miss_structure_tissue = num_miss_structure_voxels_in_subset*bx_sample_pts_volume_element

    volume_of_tissue_above_threshold_dict_for_dataframe = {"Patient ID": [patientUID],
                          "Bx ID": [bx_roi],
                          "Bx simulated bool": [bx_sim_bool],
                          "Bx type": [bx_type],
                          "Probability threshold": [probability_threshold],
                          "Volume of DIL tissue": [volume_of_dil_tissue],
                          "Miss structure roi": [structure_miss_probability_roi],
                          "Volume of structure_miss_probability_roi": [volume_of_miss_structure_tissue],
                          "Total volume": [total_structure_volume]
                          }
    
    volume_of_tissue_above_threshold_dataframe = pandas.DataFrame(volume_of_tissue_above_threshold_dict_for_dataframe)

    return volume_of_tissue_above_threshold_dataframe











def compute_sum_to_one_probabilities_by_tissue_heirarchy_with_default_tissue_for_all_false(containment_info_grand_all_structures_pandas_dataframe,
                                                        structs_referenced_dict,
                                                        default_exterior_tissue = 'Periprostatic'):

    """
    This function is designed to accept a heirachy of tissue types and then determine the number of total successes each tissue type receives
    based on the heirarchy. For example if in trial 1, the pt was contained in both the urethra and prostate, and the heirachy is dictated as [urethra,prostate]
    then the result would be urethra: 1, prostate: 0.
    """

    tissue_heirarchy_list = misc_tools.tissue_heirarchy_list_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = False
                                       )

    df = containment_info_grand_all_structures_pandas_dataframe[containment_info_grand_all_structures_pandas_dataframe['Trial num'] != 0]

    # Define the structure hierarchy
    #structure_hierarchy = ['DIL ref', 'Urethra ref', 'Rectum ref', 'OAR ref']

    # Create a categorical column for the hierarchy
    df['structure_priority'] = pandas.Categorical(df['Relative structure type'], categories=tissue_heirarchy_list, ordered=True)

    # Sort the DataFrame by Original pt index, Trial num, and structure priority
    df_sorted = df.sort_values(by=['Original pt index', 'Trial num', 'structure_priority'])

    # Flag the rows where the point is contained
    df_sorted['contained_flag'] = df_sorted['Pt contained bool'].astype(int)

    # Apply a max flag to ensure only one success per tissue type is counted, regardless of how many structures within that tissue type are True
    df_sorted['max_contained_flag'] = df_sorted.groupby(['Original pt index', 'Trial num', 'Relative structure type'])['contained_flag'].transform('max')

    # Now apply cumulative sum logic to ensure the first tissue type that wins stops others from being counted
    df_sorted['cumulative_sum'] = df_sorted.groupby(['Original pt index', 'Trial num'])['max_contained_flag'].cumsum()

    # Filter to keep only the first tissue type that wins for each trial, respecting the hierarchy
    df_filtered = df_sorted[df_sorted['cumulative_sum'] <= 1]

    # Group by Original pt index and tissue type to count unique successes per trial
    result_df = df_filtered.groupby(
        ['Original pt index', 'Relative structure type']
    ).agg({'max_contained_flag': 'sum'}).reset_index()

    # Create all possible combinations of Original pt index and Relative structure type
    all_combinations = pandas.MultiIndex.from_product(
        [df['Original pt index'].unique(), tissue_heirarchy_list],
        names=['Original pt index', 'Relative structure type']
    ).to_frame(index=False)

    # Merge the result_df with all_combinations to ensure all tissue types are included
    final_result = pandas.merge(all_combinations, result_df, on=['Original pt index', 'Relative structure type'], how='left')

    # Fill NaN values (where there was no success) with 0
    final_result['max_contained_flag'] = final_result['max_contained_flag'].fillna(0).astype(int)

    # Add default_exterior_tissue results by calculating the difference between trials and successes
    periprostatic_rows = []
    for pt_index in final_result['Original pt index'].unique():
        num_trials = df[df['Original pt index'] == pt_index]['Trial num'].nunique()
        total_successes = final_result[final_result['Original pt index'] == pt_index]['max_contained_flag'].sum()

        # Calculate the difference and add to periprostatic_rows
        periprostatic_rows.append({
            'Original pt index': pt_index,
            'Relative structure type': default_exterior_tissue,
            'max_contained_flag': num_trials - total_successes
        })

    # Convert the default_exterior_tissue rows into a DataFrame
    periprostatic_df = pandas.DataFrame(periprostatic_rows)

    # Concatenate the default_exterior_tissue rows into the final result using pandas.concat
    final_result = pandas.concat([final_result, periprostatic_df], ignore_index=True)

    # Rename the 'max_contained_flag' column to 'Total successes'
    final_result.rename(columns={'max_contained_flag': 'Total successes'}, inplace=True)

    # Ensure no NaN values in 'Total successes' column
    final_result['Total successes'] = final_result['Total successes'].fillna(0).astype(int)

    # Create a mapping from Relative structure type to Tissue class name
    tissue_class_mapping = {key: value['Tissue class name'] for key, value in structs_referenced_dict.items()}

    # Add the periprostatic to the mapping so that it doesnt get mapped to NaN
    tissue_class_mapping[default_exterior_tissue] = default_exterior_tissue

    # Apply the mapping to the 'Relative structure type' column in the final_result DataFrame
    final_result['Relative structure type'] = final_result['Relative structure type'].map(tissue_class_mapping)

    # Rename the 'Relative structure type' column to 'Tissue class'
    final_result.rename(columns={'Relative structure type': 'Tissue class'}, inplace=True)

    # Define the desired order of columns
    final_column_order = ['Tissue class', 'Original pt index', 'Total successes']

    # Reorder the DataFrame columns
    final_result = final_result[final_column_order]

    

    # Separate the nominal (0th trial) data
    nominal_df = containment_info_grand_all_structures_pandas_dataframe[containment_info_grand_all_structures_pandas_dataframe['Trial num'] == 0]

    # Apply the hierarchy logic to the nominal data
    nominal_df['structure_priority'] = pandas.Categorical(nominal_df['Relative structure type'], categories=tissue_heirarchy_list, ordered=True)
    nominal_df = nominal_df.sort_values(by=['Original pt index', 'structure_priority'])
    nominal_df['contained_flag'] = nominal_df['Pt contained bool'].astype(int)
    nominal_df['max_contained_flag'] = nominal_df.groupby(['Original pt index', 'Relative structure type'])['contained_flag'].transform('max')
    nominal_df['cumulative_sum'] = nominal_df.groupby(['Original pt index'])['max_contained_flag'].cumsum()

    # Filter to keep only the first tissue type that wins for nominal data
    nominal_filtered = nominal_df[nominal_df['cumulative_sum'] <= 1]

    # Group by Original pt index and tissue type to count unique successes for nominal
    nominal_result = nominal_filtered.groupby(['Original pt index', 'Relative structure type']).agg({'max_contained_flag': 'sum'}).reset_index()

    # Rename 'max_contained_flag' column to 'Nominal'
    nominal_result.rename(columns={'max_contained_flag': 'Nominal'}, inplace=True)

    # Merge the nominal result with the main result_df
    final_result = pandas.merge(final_result, nominal_result[['Original pt index', 'Relative structure type', 'Nominal']], 
                                on=['Original pt index', 'Relative structure type'], how='left')

    return final_result


def compute_sum_to_one_probabilities_by_tissue_heirarchy_with_default_tissue_for_all_false_and_nominal(containment_info_grand_all_structures_pandas_dataframe,
                                                        structs_referenced_dict,
                                                        default_exterior_tissue = 'Periprostatic'):
    
    """
    This function is designed to accept a heirachy of tissue types and then determine the number of total successes each tissue type receives
    based on the heirarchy. For example if in trial 1, the pt was contained in both the urethra and prostate, and the heirachy is dictated as [urethra,prostate]
    then the result would be urethra: 1, prostate: 0. It also returns the nominal value based on the 0th trial.
    """

    tissue_heirarchy_list = misc_tools.tissue_heirarchy_list_creator_func(structs_referenced_dict,
                                       append_default_exterior_tissue = False
                                       )

    df = containment_info_grand_all_structures_pandas_dataframe[containment_info_grand_all_structures_pandas_dataframe['Trial num'] != 0]

    # Define the structure hierarchy
    #structure_hierarchy = ['DIL ref', 'Urethra ref', 'Rectum ref', 'OAR ref']

    # Create a categorical column for the hierarchy
    df['structure_priority'] = pandas.Categorical(df['Relative structure type'], categories=tissue_heirarchy_list, ordered=True)

    # Sort the DataFrame by Original pt index, Trial num, and structure priority
    df_sorted = df.sort_values(by=['Original pt index', 'Trial num', 'structure_priority'])

    # Flag the rows where the point is contained
    df_sorted['contained_flag'] = df_sorted['Pt contained bool'].astype(int)

    # Apply a max flag to ensure only one success per tissue type is counted, regardless of how many structures within that tissue type are True
    df_sorted['max_contained_flag'] = df_sorted.groupby(['Original pt index', 'Trial num', 'Relative structure type'])['contained_flag'].transform('max')

    # Now apply cumulative sum logic to ensure the first tissue type that wins stops others from being counted
    df_sorted['cumulative_sum'] = df_sorted.groupby(['Original pt index', 'Trial num'])['max_contained_flag'].cumsum()

    # Filter to keep only the first tissue type that wins for each trial, respecting the hierarchy
    df_filtered = df_sorted[df_sorted['cumulative_sum'] <= 1]

    # Group by Original pt index and tissue type to count unique successes per trial
    result_df = df_filtered.groupby(
        ['Original pt index', 'Relative structure type']
    ).agg({'max_contained_flag': 'sum'}).reset_index()

    # Create all possible combinations of Original pt index and Relative structure type
    all_combinations = pandas.MultiIndex.from_product(
        [df['Original pt index'].unique(), tissue_heirarchy_list],
        names=['Original pt index', 'Relative structure type']
    ).to_frame(index=False)

    # Merge the result_df with all_combinations to ensure all tissue types are included
    final_result = pandas.merge(all_combinations, result_df, on=['Original pt index', 'Relative structure type'], how='left')

    # Fill NaN values (where there was no success) with 0
    final_result['max_contained_flag'] = final_result['max_contained_flag'].fillna(0).astype(int)

    # Add default_exterior_tissue results by calculating the difference between trials and successes
    periprostatic_rows = []
    for pt_index in final_result['Original pt index'].unique():
        num_trials = df[df['Original pt index'] == pt_index]['Trial num'].nunique()
        total_successes = final_result[final_result['Original pt index'] == pt_index]['max_contained_flag'].sum()

        # Calculate the difference and add to periprostatic_rows
        periprostatic_rows.append({
            'Original pt index': pt_index,
            'Relative structure type': default_exterior_tissue,
            'max_contained_flag': num_trials - total_successes
        })

    # Convert the default_exterior_tissue rows into a DataFrame
    periprostatic_df = pandas.DataFrame(periprostatic_rows)

    # Concatenate the default_exterior_tissue rows into the final result using pandas.concat
    final_result = pandas.concat([final_result, periprostatic_df], ignore_index=True)

    # Rename the 'max_contained_flag' column to 'Total successes'
    final_result.rename(columns={'max_contained_flag': 'Total successes'}, inplace=True)

    # Ensure no NaN values in 'Total successes' column
    final_result['Total successes'] = final_result['Total successes'].fillna(0).astype(int)

    # Create a mapping from Relative structure type to Tissue class name
    tissue_class_mapping = {key: value['Tissue class name'] for key, value in structs_referenced_dict.items()}

    # Add the periprostatic to the mapping so that it doesnt get mapped to NaN
    tissue_class_mapping[default_exterior_tissue] = default_exterior_tissue

    # Apply the mapping to the 'Relative structure type' column in the final_result DataFrame
    final_result['Relative structure type'] = final_result['Relative structure type'].map(tissue_class_mapping)

    # Rename the 'Relative structure type' column to 'Tissue class'
    final_result.rename(columns={'Relative structure type': 'Tissue class'}, inplace=True)


    # Separate the nominal (0th trial) data
    nominal_df = containment_info_grand_all_structures_pandas_dataframe[containment_info_grand_all_structures_pandas_dataframe['Trial num'] == 0]

    # Apply the hierarchy logic to the nominal data
    nominal_df['structure_priority'] = pandas.Categorical(nominal_df['Relative structure type'], categories=tissue_heirarchy_list, ordered=True)
    nominal_df = nominal_df.sort_values(by=['Original pt index', 'structure_priority'])
    nominal_df['contained_flag'] = nominal_df['Pt contained bool'].astype(int)
    nominal_df['max_contained_flag'] = nominal_df.groupby(['Original pt index', 'Relative structure type'])['contained_flag'].transform('max')
    nominal_df['cumulative_sum'] = nominal_df.groupby(['Original pt index'])['max_contained_flag'].cumsum()

    # Filter to keep only the first tissue type that wins for nominal data
    nominal_filtered = nominal_df[nominal_df['cumulative_sum'] <= 1]

    # Group by Original pt index and tissue type to count unique successes for nominal
    nominal_result = nominal_filtered.groupby(['Original pt index', 'Relative structure type']).agg({'max_contained_flag': 'sum'}).reset_index()

    # Create all possible combinations of Original pt index and Relative structure type
    all_combinations = pandas.MultiIndex.from_product(
        [df['Original pt index'].unique(), tissue_heirarchy_list],
        names=['Original pt index', 'Relative structure type']
    ).to_frame(index=False)

    # Merge the result_df with all_combinations to ensure all tissue types are included
    nominal_result = pandas.merge(all_combinations, nominal_result, on=['Original pt index', 'Relative structure type'], how='left')

    # Fill NaN values (where there was no success) with 0
    nominal_result['max_contained_flag'] = nominal_result['max_contained_flag'].fillna(0).astype(int)

    # Add default_exterior_tissue results for nominal by calculating the difference between trials and successes
    periprostatic_rows_nominal = []
    for pt_index in nominal_df['Original pt index'].unique():
        num_trials_nominal = nominal_df[nominal_df['Original pt index'] == pt_index]['Trial num'].nunique()  # Should be 1 for nominal trial
        total_successes_nominal = nominal_result[nominal_result['Original pt index'] == pt_index]['max_contained_flag'].sum()

        # Calculate the difference and add to periprostatic_rows_nominal
        periprostatic_rows_nominal.append({
            'Original pt index': pt_index,
            'Relative structure type': default_exterior_tissue,
            'max_contained_flag': num_trials_nominal - total_successes_nominal
        })

    # Convert the periprostatic rows into a DataFrame for nominal
    periprostatic_nominal_df = pandas.DataFrame(periprostatic_rows_nominal)

    # Concatenate the default_exterior_tissue rows into the nominal result
    nominal_result = pandas.concat([nominal_result, periprostatic_nominal_df], ignore_index=True)

    # Rename 'max_contained_flag' column to 'Nominal'
    nominal_result.rename(columns={'max_contained_flag': 'Nominal'}, inplace=True)

    # Apply the mapping to the 'Relative structure type' column in the final_result DataFrame
    nominal_result['Relative structure type'] = nominal_result['Relative structure type'].map(tissue_class_mapping)

    # Sort nominal_result by 'Original pt index' and 'Relative structure type'
    nominal_result_sorted = nominal_result.sort_values(by=['Original pt index', 'Relative structure type'])

    # Rename the 'Relative structure type' column to 'Tissue class'
    nominal_result_sorted.rename(columns={'Relative structure type': 'Tissue class'}, inplace=True)

    # Merge the nominal result with the main result_df
    final_result_with_nominal = pandas.merge(final_result, nominal_result_sorted[['Original pt index', 'Tissue class', 'Nominal']], 
                                on=['Original pt index', 'Tissue class'], how='left')

    # Ensure no NaN values in 'Nominal' column
    #final_result['Nominal'] = final_result['Nominal'].fillna(0).astype(int)


    # Define the desired order of columns
    final_column_order = ['Tissue class', 'Original pt index', 'Total successes', 'Nominal']

    # Reorder the DataFrame columns
    final_result_with_nominal = final_result_with_nominal[final_column_order]

    final_result_with_nominal_sorted = final_result_with_nominal.sort_values(by=['Original pt index', 'Tissue class']).reset_index(drop=True)

    return final_result_with_nominal_sorted




def prepare_2d_stacked_arr_biopsy_only_shifted_with_nominal(specific_bx_structure,
                                               num_simulations_cutoff):
    bx_only_shifted_3darr = cp.asnumpy(specific_bx_structure["MC data: bx only shifted 3darr"]) # note that the 3rd dimension slices are each MC trial
    bx_only_shifted_3darr_cutoff = bx_only_shifted_3darr[0:num_simulations_cutoff]
    unshifted_bx_sampled_pts_arr = specific_bx_structure["Random uniformly sampled volume pts arr"]
    unshifted_bx_sampled_pts_arr_3darr = np.expand_dims(unshifted_bx_sampled_pts_arr, axis=0)
    nominal_and_bx_only_shifted_3darr = np.concatenate((unshifted_bx_sampled_pts_arr_3darr,bx_only_shifted_3darr_cutoff))
    bx_only_shifted_stacked_2darr = np.reshape(nominal_and_bx_only_shifted_3darr, (-1,3) , order = 'C')

    return bx_only_shifted_stacked_2darr