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
                       stopwatch, 
                       layout_groups, 
                       master_structure_reference_dict, 
                       structs_referenced_list,
                       bx_ref,
                       oar_ref,
                       dil_ref, 
                       dose_ref,
                       plan_ref, 
                       master_structure_info_dict, 
                       biopsy_z_voxel_length, 
                       num_dose_calc_NN,
                       num_dose_NN_to_show_for_animation_plotting,
                       dose_views_jsons_paths_list,
                       containment_views_jsons_paths_list,
                       show_NN_dose_demonstration_plots,
                       show_containment_demonstration_plots,
                       biopsy_needle_compartment_length,
                       simulate_uniform_bx_shifts_due_to_bx_needle_compartment,
                       plot_uniform_shifts_to_check_plotly,
                       differential_dvh_resolution,
                       cumulative_dvh_resolution,
                       volume_DVH_percent_dose,
                       volume_DVH_quantiles_to_calculate,
                       plot_translation_vectors_pointclouds,
                       plot_cupy_containment_distribution_results,
                       plot_shifted_biopsies,
                       structure_miss_probability_roi,
                       spinner_type
                       ):
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    with live_display:
        live_display.start(refresh = True)
        num_patients = master_structure_info_dict["Global"]["Num patients"]
        num_global_structures = master_structure_info_dict["Global"]["Num structures"]
        num_MC_dose_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC dose simulations"]
        num_MC_containment_simulations = master_structure_info_dict["Global"]["MC info"]["Num MC containment simulations"]
        bx_sample_pt_lattice_spacing = master_structure_info_dict["Global"]["MC info"]["BX sample pt lattice spacing"]
        bx_sample_pts_volume_element = bx_sample_pt_lattice_spacing**3 
        
        #live_display.stop()
        max_simulations = max(num_MC_dose_simulations,num_MC_containment_simulations)
        master_structure_info_dict["Global"]["MC info"]["Max of num MC simulations"] = max_simulations

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
        testing_nominal_biopsy_containment_patient_task = patients_progress.add_task("[red]Testing nominal biopsy containment...", total=num_patients)
        testing_nominal_biopsy_containment_patient_task_completed = completed_progress.add_task("[green]Testing nominal biopsy containment", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
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
        

        #live_display.stop()
        testing_biopsy_containment_patient_task = patients_progress.add_task("[red]Testing biopsy containment (cuspatial)...", total=num_patients)
        testing_biopsy_containment_patient_task_completed = completed_progress.add_task("[green]Testing biopsy containment (cuspatial)", total=num_patients, visible = False)
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

                testing_each_non_bx_structure_containment_task = structures_progress.add_task("[cyan]~~For each non-BX structure [{}]...".format("initializing"), total=sp_patient_total_num_non_BXs)
                
                structure_shifted_bx_data_dict = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: bx and structure shifted dict"] 
                
                relative_structure_containment_results_data_frames_list = []
                for structure_info,shifted_bx_data_3darr_cp in structure_shifted_bx_data_dict.items():
                    structure_roi = structure_info[0]
                    non_bx_structure_type = structure_info[1]
                    structure_refnum = structure_info[2]
                    structure_index = structure_info[3]
                    shifted_bx_data_3darr = cp.asnumpy(shifted_bx_data_3darr_cp)
                    structures_progress.update(testing_each_non_bx_structure_containment_task, description = "[cyan]~~For each non-BX structure [{}]...".format(structure_roi))

                    # Extract and calcualte relative structure info
                    non_bx_struct_interslice_interpolation_information = master_structure_reference_dict[patientUID][non_bx_structure_type][structure_index]["Inter-slice interpolation information"]
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
                    shifted_bx_data_3darr_num_MC_containment_sims_cutoff = shifted_bx_data_3darr[0:num_MC_containment_simulations]
                    shifted_bx_data_stacked_2darr_from_all_trials_3darray = np.reshape(shifted_bx_data_3darr_num_MC_containment_sims_cutoff,(-1,3))

                    # Combine nominal and shifted
                    combined_nominal_and_shifted_bx_pts_2d_arr_XYZ = np.vstack((unshifted_bx_sampled_pts_arr,shifted_bx_data_stacked_2darr_from_all_trials_3darray))
                    combined_nominal_and_shifted_bx_pts_2d_arr_XY = combined_nominal_and_shifted_bx_pts_2d_arr_XYZ[:,0:2]
                    combined_nominal_and_shifted_bx_pts_2d_arr_Z = combined_nominal_and_shifted_bx_pts_2d_arr_XYZ[:,2]
                    
                    combined_nominal_and_shifted_nearest_interpolated_zslice_index_array, combined_nominal_and_shifted_nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input(interpolated_zvlas_list,combined_nominal_and_shifted_bx_pts_2d_arr_Z)
                    
                    combined_nominal_and_shifted_bx_data_XY_interleaved_1darr = combined_nominal_and_shifted_bx_pts_2d_arr_XY.flatten()
                    combined_nominal_and_shifted_bx_data_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(combined_nominal_and_shifted_bx_data_XY_interleaved_1darr)

                    
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

                    relative_structure_containment_results_data_frames_list.append(containment_info_grand_cudf_dataframe)

                    if (show_containment_demonstration_plots == True) & (non_bx_structure_type == 'DIL ref'):
                        for trial_num, single_trial_shifted_bx_data_arr in enumerate(shifted_bx_data_3darr_num_MC_containment_sims_cutoff):
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
                    
                    structures_progress.update(testing_each_non_bx_structure_containment_task, advance=1)
                
                structures_progress.remove_task(testing_each_non_bx_structure_containment_task)
                
                # concatenate containment results into a single dataframe
                containment_info_grand_all_structures_cudf_dataframe = cudf.concat(relative_structure_containment_results_data_frames_list, ignore_index=True)
                # free up GPU memory
                del containment_info_grand_cudf_dataframe
                del relative_structure_containment_results_data_frames_list
                # Update the master dictionary
                master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: MC sim containment raw results dataframe"] = containment_info_grand_all_structures_cudf_dataframe.to_pandas()

                biopsies_progress.update(testing_biopsy_containment_task, advance=1)
            biopsies_progress.remove_task(testing_biopsy_containment_task)

            patients_progress.update(testing_biopsy_containment_patient_task, advance = 1)
            completed_progress.update(testing_biopsy_containment_patient_task_completed, advance = 1)
        patients_progress.update(testing_biopsy_containment_patient_task, visible = False)
        completed_progress.update(testing_biopsy_containment_patient_task_completed, visible = True)
        live_display.refresh()

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
                
                sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
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
                    """
                    st = time.time()
                    bx_containment_counter_by_org_pt_ind_list = [len(containment_info_grand_all_structures_cudf_dataframe[(containment_info_grand_all_structures_cudf_dataframe["Relative structure ROI"] == structure_roi)  
                                                                                                           & (containment_info_grand_all_structures_cudf_dataframe["Relative structure index"] == structure_index)
                                                                                                           & (containment_info_grand_all_structures_cudf_dataframe["Pt contained bool"] == True)
                                                                                                           & (containment_info_grand_all_structures_cudf_dataframe["Original pt index"] == pt_index)
                                                                                                           ])
                                                                                                           for pt_index in range(num_sample_pts_in_bx)
                                                                                                           ]
                    et = time.time()
                    print("Org: "+str(et-st))
                    """
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

        
        calc_MC_stat_biopsy_containment_task = patients_progress.add_task("[red]Calculating MC statistics [{}]...".format("initializing"), total=num_patients)
        calc_MC_stat_biopsy_containment_task_complete = completed_progress.add_task("[green]Calculating MC statistics", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(calc_MC_stat_biopsy_containment_task, description = "[red]Calculating MC statistics [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
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
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
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
                specific_bx_structure_tumor_tissue_dict["Tumor tissue nominal arr"] = bx_nominal_containment_dil_exclusive_by_org_pt_ind_arr

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
                probabilities_CI_arr = mf.confidence_intervals_95_from_calculated_SE(probability_pt_wise_dil_tissue_arr, probabilities_standard_err_arr)
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
                
                biopsies_progress.update(calc_mutual_probabilities_stat_each_bx_structure_containment_task, advance = 1)
            biopsies_progress.remove_task(calc_mutual_probabilities_stat_each_bx_structure_containment_task)
            patients_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task, advance = 1)
            completed_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task_complete, advance = 1)
        patients_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task, visible = False)
        completed_progress.update(calc_mutual_probabilities_stat_biopsy_containment_task_complete,visible = True)
        live_display.refresh()
        
        
        # voxelize containment results
        biopsy_voxelize_containment_task = patients_progress.add_task("[red]Voxelizing containment results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_containment_task_complete = completed_progress.add_task("[green]Voxelizing containment results", total=num_patients)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing containment results [{}]...".format(patientUID))
            structure_organized_for_bx_data_blank_dict = create_patient_specific_structure_dict_for_data(pydicom_item,structs_referenced_list)
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
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
                """
                dosimetric_localization_all_MC_trials_list = dosimetric_localization_parallel(parallel_pool, 
                                                                                              nominal_and_bx_only_shifted_3darr, 
                                                                                              specific_bx_structure, 
                                                                                              dose_ref_dict, 
                                                                                              dose_ref, 
                                                                                              phys_space_dose_map_phys_coords_2d_arr, 
                                                                                              phys_space_dose_map_dose_2d_arr, 
                                                                                              num_dose_calc_NN)
                """
                
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
                """
                dosimetric_localization_nominal_NN_parent_obj = dosimetric_localization_nominal_and_all_MC_trials_list[0]
                dosimetric_localization_all_MC_trials_list = dosimetric_localization_nominal_and_all_MC_trials_list[1:]
                """

                # MC trials
                dosimetric_localization_nominal_and_all_MC_trials_list_NN_lists_only = [NN_parent_obj.NN_data_list for NN_parent_obj in dosimetric_localization_nominal_and_all_MC_trials_list]
                dosimetric_localization_NN_child_objs_by_bx_point_nominal_and_all_trials_list = list(zip(*dosimetric_localization_nominal_and_all_MC_trials_list_NN_lists_only))
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_list = [[NN_child_obj.nearest_dose for NN_child_obj in fixed_bx_pt_NN_objs_list] for fixed_bx_pt_NN_objs_list in dosimetric_localization_NN_child_objs_by_bx_point_nominal_and_all_trials_list]
                dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr = np.array(dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_list)
                # Nominal
                """
                dosimetric_localization_NN_child_objs_by_bx_point_nominal_list = dosimetric_localization_nominal_NN_parent_obj.NN_data_list
                dosimetric_localization_dose_vals_by_bx_point_nominal_list = [nn_dose_child_obj.nearest_dose for nn_dose_child_obj in dosimetric_localization_NN_child_objs_by_bx_point_nominal_list]
                """

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
                for vol_dose_percent in volume_DVH_percent_dose:
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)] = {"Nominal": None, 
                                                                               "All MC trials list": [], 
                                                                               "Mean": None, 
                                                                               "STD": None, 
                                                                               "Quantiles": None
                                                                               }
                    
                for trial_index in range(num_nominal_and_all_dose_trials):
                    dosimetric_localization_dose_vals_all_pts_specific_MC_trial = dosimetric_localization_dose_vals_by_bx_point_nominal_and_all_trials_arr[:,trial_index]
                    differential_dvh_histogram_counts_specific_MC_trial, differential_dvh_histogram_edges_specific_MC_trial = np.histogram(dosimetric_localization_dose_vals_all_pts_specific_MC_trial, bins = differential_dvh_resolution, range = differential_dvh_range)
                    differential_dvh_histogram_counts_by_MC_trial_arr[trial_index,:] = differential_dvh_histogram_counts_specific_MC_trial
                    differential_dvh_histogram_edges_by_MC_trial_arr[trial_index,:] = differential_dvh_histogram_edges_specific_MC_trial
                    
                
                    # find specific DVH metrics for nominal and all MC trials
                    for vol_dose_percent in volume_DVH_percent_dose:
                        dose_threshold = (vol_dose_percent/100)*ctv_dose
                        counts_for_vol_dose_percent = dosimetric_localization_dose_vals_all_pts_specific_MC_trial[dosimetric_localization_dose_vals_all_pts_specific_MC_trial > dose_threshold].shape[0]
                        percent_for_vol_dose_percent = (counts_for_vol_dose_percent/num_sampled_bx_pts)*100
                        if trial_index == 0:
                            dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Nominal"] = percent_for_vol_dose_percent
                        else:
                            dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["All MC trials list"].append(percent_for_vol_dose_percent)

                
                for vol_dose_percent in volume_DVH_percent_dose:
                    dvh_metric_vol_dose_percent_MC_trials_only_list = dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["All MC trials list"]
                    dvh_metric_all_trials_arr = np.array(dvh_metric_vol_dose_percent_MC_trials_only_list) 
                    mean_of_dvh_metric = np.mean(dvh_metric_all_trials_arr)
                    std_of_dvh_metric = np.std(dvh_metric_all_trials_arr)
                    quantiles_of_dvh_metric = {'Q'+str(q): np.quantile(dvh_metric_all_trials_arr, q/100) for q in volume_DVH_quantiles_to_calculate}
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Mean"] = mean_of_dvh_metric
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["STD"] = std_of_dvh_metric
                    dvh_metric_vol_dose_percent_dict[str(vol_dose_percent)]["Quantiles"] = quantiles_of_dvh_metric


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



        computing_MLE_statistics_dose_task = patients_progress.add_task("[red]Computing dosimetric localization statistics (MLE) [{}]...".format("initializing"), total=num_patients)
        computing_MLE_statistics_dose_task_complete = completed_progress.add_task("[green]Computing dosimetric localization statistics (MLE)", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(computing_MLE_statistics_dose_task, description = "[red]Computing dosimetric localization statistics (MLE) [{}]...".format(patientUID))
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            compile_results_dose_NN_biopsy_containment_by_biopsy_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total = sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_structure_roi = specific_bx_structure["ROI"]
                biopsies_progress.update(compile_results_dose_NN_biopsy_containment_by_biopsy_task, description = "[cyan]~For each biopsy [{}]...".format(specific_bx_structure_roi))
                dosimetric_localization_dose_vals_by_bx_point_all_trials_arr = specific_bx_structure["MC data: Dose vals for each sampled bx pt arr (all MC trials)"] 
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



        # voxelize dose results
        biopsy_voxelize_dose_task = patients_progress.add_task("[red]Voxelizing dose results [{}]...".format("initializing"), total=num_patients)
        biopsy_voxelize_dose_task_complete = completed_progress.add_task("[green]Voxelizing dose results", total=num_patients, visible = False)
        for patientUID,pydicom_item in master_structure_reference_dict.items():
            patients_progress.update(biopsy_voxelize_containment_task, description = "[red]Voxelizing dose results [{}]...".format(patientUID))
            
            sp_patient_total_num_structs = master_structure_info_dict["By patient"][patientUID]["All ref"]["Total num structs"]
            sp_patient_total_num_BXs = master_structure_info_dict["By patient"][patientUID][bx_ref]["Num structs"]
            sp_patient_total_num_non_BXs = sp_patient_total_num_structs - sp_patient_total_num_BXs

            biopsy_voxelize_each_bx_structure_dose_task = biopsies_progress.add_task("[cyan]~For each biopsy [{}]...".format("initializing"), total=sp_patient_total_num_BXs)
            for specific_bx_structure_index, specific_bx_structure in enumerate(pydicom_item[bx_ref]):
                specific_bx_dose_results_arr = master_structure_reference_dict[patientUID][bx_ref][specific_bx_structure_index]["MC data: Dose vals for each sampled bx pt arr (all MC trials)"] 
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
        

        return master_structure_reference_dict, live_display


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