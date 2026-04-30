import numpy as np
import math
from shapely.geometry import Polygon
from line_profiler import LineProfiler
import point_containment_tools
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import plotting_funcs
import MC_simulator_convex
import misc_tools
from . import biopsy_optimizer_module_v1_helpers
import dataframe_builders
import dataframe_dtype_policy
import pandas
import plotly.graph_objects as go


def biopsy_optimizer_module_v1(master_structure_reference_dict,
                              master_structure_info_dict,
                              structs_referenced_dict,
                              bx_ref,
                              dil_ref,
                              oar_ref,
                              all_ref_key,
                              voxel_size_for_dil_optimizer_grid,
                              optimal_normal_dist_option,
                              bias_LR_multiplier,
                              bias_AP_multiplier,
                              bias_SI_multiplier,
                              num_normal_dist_points_for_biopsy_optimizer,
                              normal_dist_sigma_factor_biopsy_optimizer,
                              plot_each_normal_dist_containment_result_bool,
                              plot_optimization_point_lattice_bool,
                              show_optimization_point_bool,
                              cupy_array_upper_limit_NxN_size_input,
                              numpy_array_upper_limit_NxN_size_input,
                              nearest_zslice_vals_and_indices_cupy_generic_max_size,
                              nearest_zslice_vals_and_indices_numpy_generic_max_size,
                              constant_z_slice_polygons_handler_option,
                              remove_consecutive_duplicate_points_in_polygons,
                              include_edges_in_log_files,
                              custom_cuda_kernel_type,
                              demonstrate_dil_optimization_points_inside_correctness_bool_1,
                              demonstrate_dil_optimization_points_inside_correctness_bool_2,
                              demonstrate_dil_optimization_points_inside_correctness_num_3,
                              generate_cuda_log_files_biopsy_optimizer,
                              display_optimization_contour_plots_bool,
                              layout_groups,
                              patients_progress,
                              structures_progress,
                              indeterminate_progress_sub,
                              important_info,
                              completed_progress,
                              live_display,
                              ):


    #live_display.stop()

    patientUID_default = "Initializing"
    processing_patients_task_main_description = "[red]Optimizing Bx location within DILs [{}]...".format(patientUID_default)
    processing_patients_task_completed_main_description = "[green]Optimizing Bx location within DILs"
    processing_patients_task = patients_progress.add_task(processing_patients_task_main_description, total=master_structure_info_dict["Global"]["Num cases"])
    processing_patients_task_completed = completed_progress.add_task(processing_patients_task_completed_main_description, total=master_structure_info_dict["Global"]["Num cases"], visible = False)

    for patientUID,pydicom_item in master_structure_reference_dict.items():
        processing_patients_task_main_description = "[red]Optimizing Bx location within DILs [{}]...".format(patientUID)
        patients_progress.update(processing_patients_task, description = processing_patients_task_main_description)
        #####
        
        ### SELECT PROSTATE, OR DEFAULT TO ORIGIN FOR PROSTATE COM IF NONE FOUND
        # selected_prostate_info, message_string, prostate_found_bool, num_prostates_found = misc_tools.specific_structure_selector(pydicom_item,
        #                         oar_ref,
        #                         prostate_contour_name)    

        sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

        specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
        selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

        prostate_ID = selected_prostate_info["Structure ID"]
        prostate_ref_type = selected_prostate_info["Struct ref type"]
        prostate_ref_num = selected_prostate_info["Dicom ref num"]
        prostate_structure_index = selected_prostate_info["Index number"]
        prostate_found_bool = selected_prostate_info["Struct found bool"]


        if prostate_found_bool == True:
            prostate_centroid = pydicom_item[prostate_ref_type][prostate_structure_index]["Structure global centroid"].reshape(3)
        else: 
            important_info.add_text_line('Prostate not found! Defaulting prostate centroid to Zero-vector')
            prostate_centroid = np.array([0,0,0])



        ## GENERATE LATTICE ENCOMPASSING ALL GEOMETRIES
        # add the dils!
        list_of_all_dils_interpolated_pts = []
        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
            sp_dil_interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
            sp_dil_interpolated_pts_np_arr = sp_dil_interslice_interpolation_information.interpolated_pts_np_arr
            list_of_all_dils_interpolated_pts.append(sp_dil_interpolated_pts_np_arr)
        #all_dils_interpolated_pts = np.vstack(list_of_all_dils_interpolated_pts)

        # add the OARs!
        list_of_all_oar_interpolated_pts = []
        for specific_oar_structure in pydicom_item[oar_ref]:
            oar_interslice_interpolation_information = specific_oar_structure["Inter-slice interpolation information"]
            oar_interpolated_pts_np_arr = oar_interslice_interpolation_information.interpolated_pts_np_arr
            list_of_all_oar_interpolated_pts.append(oar_interpolated_pts_np_arr)

        all_geometries_list_of_interpolated_pts = list_of_all_dils_interpolated_pts + list_of_all_oar_interpolated_pts
        all_geometries_interpolated_pts = np.vstack(all_geometries_list_of_interpolated_pts)

        # Before, only added prostate, but better to add ALL OARs
        """
        if prostate_found_bool == True:
            prostate_interslice_interpolation_information = pydicom_item[prostate_ref_type][prostate_structure_index]["Inter-slice interpolation information"]
            prostate_interpolated_pts_np_arr = prostate_interslice_interpolation_information.interpolated_pts_np_arr
            all_geometries_interpolated_pts = np.vstack([all_dils_interpolated_pts,prostate_interpolated_pts_np_arr])
        else: 
            all_geometries_interpolated_pts = all_dils_interpolated_pts
        """

        # all geometries means dils + prostate (if a prostate could be found!)
        all_geometries_interpolated_pts_point_cloud = point_containment_tools.create_point_cloud(all_geometries_interpolated_pts)
        interpolated_pts_point_cloud_color = np.array([0,0,1])
        all_geometries_interpolated_pts_point_cloud.paint_uniform_color(interpolated_pts_point_cloud_color)

        all_geometries_axis_aligned_bounding_box = all_geometries_interpolated_pts_point_cloud.get_axis_aligned_bounding_box()
        all_geometries_axis_aligned_bounding_box_points_arr = np.asarray(all_geometries_axis_aligned_bounding_box.get_box_points())
        all_geometries_bounding_box_color = np.array([0,0,0], dtype=float)
        all_geometries_axis_aligned_bounding_box.color = all_geometries_bounding_box_color
        all_geometries_max_bounds = np.amax(all_geometries_axis_aligned_bounding_box_points_arr, axis=0)
        all_geometries_min_bounds = np.amin(all_geometries_axis_aligned_bounding_box_points_arr, axis=0)

        lattice_sizex = int(math.ceil(abs(all_geometries_max_bounds[0]-all_geometries_min_bounds[0])/voxel_size_for_dil_optimizer_grid) + 1)
        lattice_sizey = int(math.ceil(abs(all_geometries_max_bounds[1]-all_geometries_min_bounds[1])/voxel_size_for_dil_optimizer_grid) + 1)
        lattice_sizez = int(math.ceil(abs(all_geometries_max_bounds[2]-all_geometries_min_bounds[2])/voxel_size_for_dil_optimizer_grid) + 1)
        origin = all_geometries_min_bounds

        # generate cubic lattice of points
        all_geometries_centered_cubic_lattice_arr = MC_simulator_convex.generate_cubic_lattice(voxel_size_for_dil_optimizer_grid, 
                                                                                            lattice_sizex,
                                                                                            lattice_sizey,
                                                                                            lattice_sizez,
                                                                                            origin)


        # CREATE A COPY TO REMOVE THE POINTS CONTAINED IN THE DILS!
        #all_geometries_centered_cubic_lattice_with_dil_points_removed_arr = all_geometries_centered_cubic_lattice_arr.copy()
        

        # Create empty dataframe for all contained points in lattice
        #dil_contained_points_df = pandas.DataFrame()
        
        # Make an empty array to keep track of all the points 
        #centered_cubic_lattice_points_only_contained_in_ANY_dil_arr = np.empty((0, 3))

        #####
        structureID_default = "Initializing"
        num_dil_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][dil_ref]["Num structs"]
        processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_default)
        processing_structures_task = structures_progress.add_task(processing_structures_task_main_description, total=num_dil_structs_patient_specific)
        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
            structureID_dil = specific_dil_structure["ROI"]
            structure_reference_number_dil = specific_dil_structure["Ref #"]
            processing_structures_task_main_description = "[cyan]Processing structures [{},{}]...".format(patientUID,structureID_dil)
            structures_progress.update(processing_structures_task, description = processing_structures_task_main_description)

            ### FIND OPTIMAL POSITION FOR BIOPSY SAMPLING (DIL ONLY)
            structure_info = misc_tools.specific_structure_info_dict_creator('given', specific_structure = specific_dil_structure)

            interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]
            interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
            interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
            zslices_list = interslice_interpolation_information.interpolated_pts_list
            # Extract the dil centroid
            dil_global_centroid = specific_dil_structure["Structure global centroid"]


            ### OLD METHODOLOGY FOR REMOVING POINTS OUTSIDE OF DIL
            """
            pr = cProfile.Profile()
            pr.enable()
            # create geoseries of the dil structure for containment tests
            max_zval = max(interpolated_zvals_list)
            min_zval = min(interpolated_zvals_list)
            zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in zslices_list]
            zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(zslices_polygons_list))

            

            ### CONSTRUCT THE LATTICE POINTS TO PASS TO THE OPTIMIZER FUNCTION
            all_geometries_centered_cubic_lattice_arr_XY = all_geometries_centered_cubic_lattice_arr[:,0:2]
            all_geometries_centered_cubic_lattice_arr_Z = all_geometries_centered_cubic_lattice_arr[:,2]

            
            #nearest_interpolated_zslice_for_test_lattice_index_array, nearest_interpolated_zslice_for_test_lattice_vals_array = point_containment_tools.take_closest_cupy(interpolated_zvals_list, all_geometries_centered_cubic_lattice_arr_Z)
            
            nearest_interpolated_zslice_for_test_lattice_index_array, nearest_interpolated_zslice_for_test_lattice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvals_list, 
                                                                                                                                                all_geometries_centered_cubic_lattice_arr_Z,
                                                                                                                                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                structures_progress
                                                                                                                                                )
            
            
            all_geometries_centered_cubic_lattice_XY_interleaved_1darr = all_geometries_centered_cubic_lattice_arr_XY.flatten()
            all_geometries_centered_cubic_lattice_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(all_geometries_centered_cubic_lattice_XY_interleaved_1darr)

            
            # Test point containment to remove points from the potential optimization testing point lattice that are not inside the DIL
            containment_info_for_all_lattice_points_grand_pandas_dataframe, live_display = point_containment_tools.cuspatial_points_contained_generic_numpy_pandas(zslices_polygons_cuspatial_geoseries,
                all_geometries_centered_cubic_lattice_XY_cuspatial_geoseries_points, 
                all_geometries_centered_cubic_lattice_arr, 
                nearest_interpolated_zslice_for_test_lattice_index_array,
                nearest_interpolated_zslice_for_test_lattice_vals_array,
                max_zval,
                min_zval,
                structure_info,
                layout_groups,
                live_display,
                structures_progress,
                upper_limit_size_input = cupy_array_upper_limit_NxN_size_input,
                )
            del nearest_interpolated_zslice_for_test_lattice_index_array
            del nearest_interpolated_zslice_for_test_lattice_vals_array
            live_display.refresh()


            # Print profiling results
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            ps.print_stats()
            print(s.getvalue())

            ### OLD METHODOLOGY FOR REMOVING POINTS OUTSIDE OF DIL END 
            

            """



            
            ### NEW METHODOLOGY FOR REMOVING POINTS OUTSIDE OF DIL
            #pr = cProfile.Profile()
            #pr.enable()

            # maps the first test structure to the first relative structure (since there is only 1 test structure and 1 relative structure)              
            test_struct_to_relative_struct_1d_mapping_array = np.array([0])          
            log_sub_dirs_list = [patientUID, structureID_dil]
            if generate_cuda_log_files_biopsy_optimizer == True:
                custom_cuda_log_file_name = "cuda_dil_bioposy_optimization_lattice.txt"
            else:
                custom_cuda_log_file_name = None 


            containment_result_for_all_lattice_points_cp_arr, prepper_output_tuple = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function([zslices_list],
                                all_geometries_centered_cubic_lattice_arr[np.newaxis,:,:],
                                test_struct_to_relative_struct_1d_mapping_array,
                                constant_z_slice_polygons_handler_option = constant_z_slice_polygons_handler_option,
                                remove_consecutive_duplicate_points_in_polygons = remove_consecutive_duplicate_points_in_polygons,
                                log_sub_dirs_list = log_sub_dirs_list,
                                log_file_name = custom_cuda_log_file_name,
                                include_edges_in_log = include_edges_in_log_files,
                                kernel_type = custom_cuda_kernel_type)

                                

            # For line profiling purposes
            """
            lp = LineProfiler()
            lp.add_function(custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function)  # Add another function to compare
            lp.add_function(custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.one_to_one_point_in_polygon_cupy_arr_version)
            lp_wrapper = lp(custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function)

            lp.enable()

            # Now call the function through the wrapper
            containment_result_for_all_lattice_points_cp_arr, prepper_output_tuple = lp_wrapper(
                [zslices_list],
                all_geometries_centered_cubic_lattice_arr[np.newaxis, :, :],
                test_struct_to_relative_struct_1d_mapping_array, 
                constant_z_slice_polygons_handler_option=constant_z_slice_polygons_handler_option,
                remove_consecutive_duplicate_points_in_polygons=remove_consecutive_duplicate_points_in_polygons,
                log_sub_dirs_list=log_sub_dirs_list,
                log_file_name=custom_cuda_log_file_name,
                include_edges_in_log=include_edges_in_log_files,
                kernel_type=custom_cuda_kernel_type
            )
            lp.disable()
            lp.print_stats()
            input("Press enter to continue")
            """

            # old methodology for calling custom kernel, now use mother function to handle all the steps
            """
            nearest_zslice_index_and_values_3d_arr, all_structures_list_of_2d_arr, all_structures_slices_indices_list = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_arr_version_prepper([zslices_list],
                                                all_geometries_centered_cubic_lattice_arr[np.newaxis,:,:],
                                                test_struct_to_relative_struct_1d_mapping_array,
                                                constant_z_slice_polygons_handler_option = 'auto-close-if-open')


            log_sub_dirs_list = [patientUID, structureID_dil]
            custom_cuda_log_file_name = "cuda_dil_bioposy_optimization.txt"
            containment_result_for_all_lattice_points_cp_arr = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.test_points_against_polygons_cupy_3d_arr_version(nearest_zslice_index_and_values_3d_arr, 
                                                                                            all_geometries_centered_cubic_lattice_arr[np.newaxis,:,:], 
                                                                                            all_structures_list_of_2d_arr, 
                                                                                            all_structures_slices_indices_list,
                                                                                            log_sub_dirs_list = log_sub_dirs_list, 
                                                                                            log_file_name = custom_cuda_log_file_name,
                                                                                            include_edges_in_log = include_edges_in_log_files,
                                                                                            kernel_type=custom_cuda_kernel_type)
            """
            
            containment_info_for_all_lattice_points_grand_pandas_dataframe = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.create_containment_results_dataframe_type_2I(structure_info, 
                                                                                                                    prepper_output_tuple[0], 
                                                                                                                    all_geometries_centered_cubic_lattice_arr[np.newaxis,:,:], 
                                                                                                                    containment_result_for_all_lattice_points_cp_arr,
                                                                                                                    do_not_convert_column_names_to_categorical = ["Pt contained bool"],
                                                                                                                    float_dtype = np.float32,
                                                                                                                    int_dtype = np.int32)
            #pr.disable()

            # Print profiling results
            #s = io.StringIO()
            #ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            #ps.print_stats()
            #print(s.getvalue())

            ### NEW METHODOLOGY FOR REMOVING POINTS OUTSIDE OF DIL END
            



            ### demonstrate correctness
            if demonstrate_dil_optimization_points_inside_correctness_bool_1 == True:

                plotting_funcs.plot_containment_info_dataframe_to_point_cloud_plus_other_clouds(containment_info_for_all_lattice_points_grand_pandas_dataframe, 
                            "Test pt X", 
                            "Test pt Y", 
                            "Test pt Z",
                            "Pt clr R",
                            "Pt clr G",
                            "Pt clr B",
                            additional_point_clouds=[specific_dil_structure['Interpolated structure point cloud dict']['Full with end caps']])


            #containment_info_for_all_lattice_points_grand_pandas_dataframe = containment_info_for_all_lattice_points_grand_cudf_dataframe.to_pandas()
            
            containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe = containment_info_for_all_lattice_points_grand_pandas_dataframe.drop(containment_info_for_all_lattice_points_grand_pandas_dataframe[containment_info_for_all_lattice_points_grand_pandas_dataframe["Pt contained bool"] == False].index).reset_index()
            containment_info_for_lattice_points_NOT_in_sp_dil_grand_pandas_dataframe = containment_info_for_all_lattice_points_grand_pandas_dataframe.drop(containment_info_for_all_lattice_points_grand_pandas_dataframe[containment_info_for_all_lattice_points_grand_pandas_dataframe["Pt contained bool"] == True].index).reset_index()
            del containment_info_for_all_lattice_points_grand_pandas_dataframe

            #dil_contained_points_df = dil_contained_points_df.append(containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe)

            centered_cubic_lattice_points_contained_only_in_sp_dil_arr = all_geometries_centered_cubic_lattice_arr[containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe["index"].to_numpy()]
            centered_cubic_lattice_points_NOT_contained_only_in_sp_dil_arr = all_geometries_centered_cubic_lattice_arr[containment_info_for_lattice_points_NOT_in_sp_dil_grand_pandas_dataframe["index"].to_numpy()]
            #centered_cubic_lattice_points_NOT_contained_in_ANY_dil_arr = centered_cubic_lattice_points_NOT_contained_in_ANY_dil_arr[containment_info_for_lattice_points_NOT_in_sp_dil_grand_pandas_dataframe["index"].to_numpy()]
            del containment_info_for_lattice_points_in_sp_dil_grand_pandas_dataframe

            optimal_locations_dataframe, potential_optimal_locations_dataframe, zero_locations_dataframe, live_display = biopsy_optimizer_module_v1_helpers.find_dil_optimal_sampling_position(specific_dil_structure,
                                                                                            optimal_normal_dist_option,
                                                                                            bias_LR_multiplier,
                                                                                            bias_AP_multiplier,
                                                                                            bias_SI_multiplier,
                                                                                            patientUID,
                                                                                            structs_referenced_dict,
                                                                                            bx_ref,
                                                                                            dil_ref,
                                                                                            interpolated_pts_np_arr,
                                                                                            interpolated_zvals_list,
                                                                                            zslices_list,
                                                                                            structure_info,
                                                                                            dil_global_centroid,
                                                                                            voxel_size_for_dil_optimizer_grid,
                                                                                            num_normal_dist_points_for_biopsy_optimizer,
                                                                                            normal_dist_sigma_factor_biopsy_optimizer,
                                                                                            prostate_centroid,
                                                                                            selected_prostate_info,
                                                                                            plot_each_normal_dist_containment_result_bool,
                                                                                            plot_optimization_point_lattice_bool,
                                                                                            show_optimization_point_bool,
                                                                                            layout_groups,
                                                                                            live_display,
                                                                                            cupy_array_upper_limit_NxN_size_input,
                                                                                            numpy_array_upper_limit_NxN_size_input,
                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                            nearest_zslice_vals_and_indices_numpy_generic_max_size,
                                                                                            structures_progress,
                                                                                            constant_z_slice_polygons_handler_option,
                                                                                            remove_consecutive_duplicate_points_in_polygons,
                                                                                            include_edges_in_log_files,
                                                                                            custom_cuda_kernel_type,
                                                                                            demonstrate_dil_optimization_points_inside_correctness_bool_2,
                                                                                            demonstrate_dil_optimization_points_inside_correctness_num_3,
                                                                                            generate_cuda_log_files_biopsy_optimizer,
                                                                                            test_lattice_arr = centered_cubic_lattice_points_contained_only_in_sp_dil_arr,
                                                                                            all_points_to_set_to_zero_arr = centered_cubic_lattice_points_NOT_contained_only_in_sp_dil_arr # This was added to make the "search" volume to include the entire volume
                                                                                            )


            # add constant plane indices 

            potential_optimal_locations_dataframe = misc_tools.assign_plane_indices(potential_optimal_locations_dataframe, 
                                    voxel_size_for_dil_optimizer_grid, 
                                    'Test location (Prostate centroid origin) (X)',
                                    'Test location (Prostate centroid origin) (Y)',
                                    'Test location (Prostate centroid origin) (Z)')
            
            zero_locations_dataframe = misc_tools.assign_plane_indices(zero_locations_dataframe, 
                                    voxel_size_for_dil_optimizer_grid, 
                                    'Test location (Prostate centroid origin) (X)',
                                    'Test location (Prostate centroid origin) (Y)',
                                    'Test location (Prostate centroid origin) (Z)')

            optimal_locations_dataframe = misc_tools.assign_plane_indices(optimal_locations_dataframe, 
                                    voxel_size_for_dil_optimizer_grid, 
                                    'Test location (Prostate centroid origin) (X)',
                                    'Test location (Prostate centroid origin) (Y)',
                                    'Test location (Prostate centroid origin) (Z)')



            #del centered_cubic_lattice_points_contained_only_in_sp_dil_arr
            live_display.refresh()
            
            # Save the dil centroid optimization result in a seperate dataframe 
            dil_centroids_optimization_locations_dataframe = pandas.DataFrame(potential_optimal_locations_dataframe.loc[[0],:])

            dil_centroids_optimization_locations_dataframe = misc_tools.assign_plane_indices(dil_centroids_optimization_locations_dataframe, 
                                    voxel_size_for_dil_optimizer_grid, 
                                    'Test location (Prostate centroid origin) (X)',
                                    'Test location (Prostate centroid origin) (Y)',
                                    'Test location (Prostate centroid origin) (Z)')
            
            dil_centroids_optimization_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
                dil_centroids_optimization_locations_dataframe,
                threshold=0.25,
                ignore_types=(np.floating,),
                do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
            )
            optimal_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
                optimal_locations_dataframe,
                threshold=0.25,
                ignore_types=(np.floating,),
                do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
            )
            potential_optimal_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
                potential_optimal_locations_dataframe,
                threshold=0.25,
                ignore_types=(np.floating,),
                do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
            )
            zero_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
                zero_locations_dataframe,
                threshold=0.25,
                ignore_types=(np.floating,),
                do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
            )

            #potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])
            #centered_cubic_lattice_points_only_contained_in_ANY_dil_arr = np.vstack([centered_cubic_lattice_points_only_contained_in_ANY_dil_arr,potential_optimal_locations_dataframe_centroid_dropped])


            specific_dil_structure["Biopsy optimization: DIL centroid optimal biopsy location dataframe"] = dil_centroids_optimization_locations_dataframe
            specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"] = optimal_locations_dataframe
            specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"] = potential_optimal_locations_dataframe
            specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"] = zero_locations_dataframe
            specific_dil_structure["Biopsy optimization: cubic lattice of optimization points only in dil"] = centered_cubic_lattice_points_contained_only_in_sp_dil_arr

            structures_progress.update(processing_structures_task, advance=1)
        structures_progress.update(processing_structures_task, visible = False)
        
        ###
        indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Tying results in a bow", total = None)
        ###
        
        """
        centered_cubic_lattice_points_NOT_contained_in_dils_arr = np.delete(all_geometries_centered_cubic_lattice_arr,dil_contained_points_df["index"].to_numpy(),axis = 0)
        #centered_cubic_lattice_points_NOT_contained_in_dils_arr = all_geometries_centered_cubic_lattice_arr[dil_contained_points_df["index"].to_numpy()]
        del dil_contained_points_df


        num_lattice_points_not_in_dils = centered_cubic_lattice_points_NOT_contained_in_dils_arr.shape[0]

        # Calculate test lattice in prostate coordinates
        prostate_centroid_to_test_location_arr = centered_cubic_lattice_points_NOT_contained_in_dils_arr - prostate_centroid
        distance_to_prostate_centroid_arr = np.linalg.norm(prostate_centroid_to_test_location_arr, axis=1) 

        centered_cubic_lattice_points_NOT_contained_in_dils_dict_for_dataframe = {"Patient ID": [patientUID]*num_lattice_points_not_in_dils,
                                            'Test location vector': list(centered_cubic_lattice_points_NOT_contained_in_dils_arr),
                                            'Test location (X)': centered_cubic_lattice_points_NOT_contained_in_dils_arr[:,0],
                                            'Test location (Y)': centered_cubic_lattice_points_NOT_contained_in_dils_arr[:,1],
                                            'Test location (Z)': centered_cubic_lattice_points_NOT_contained_in_dils_arr[:,2],
                                            'Selected prostate ROI': [selected_prostate_info["Structure ID"]]*num_lattice_points_not_in_dils,
                                            'Selected prostate type': [selected_prostate_info["Struct ref type"]]*num_lattice_points_not_in_dils,
                                            'Selected prostate ref num': [selected_prostate_info["Dicom ref num"]]*num_lattice_points_not_in_dils,
                                            'Selected prostate index': [selected_prostate_info["Index number"]]*num_lattice_points_not_in_dils,
                                            'Test location vector (Prostate centroid origin)': list(prostate_centroid_to_test_location_arr),
                                            'Test location (Prostate centroid origin) (X)': prostate_centroid_to_test_location_arr[:,0],
                                            'Test location (Prostate centroid origin) (Y)': prostate_centroid_to_test_location_arr[:,1],
                                            'Test location (Prostate centroid origin) (Z)': prostate_centroid_to_test_location_arr[:,2],
                                            'Dist to Prostate centroid': distance_to_prostate_centroid_arr,
                                            'Number of normal dist points contained': [0]*num_lattice_points_not_in_dils,
                                            'Number of normal dist points tested': [num_normal_dist_points_for_biopsy_optimizer]*num_lattice_points_not_in_dils,
                                            'Proportion of normal dist points contained': [0]*num_lattice_points_not_in_dils
                                            }
        # all points not in the dil, these values were set to zero, this is for the contour plot production!
        centered_cubic_lattice_non_dil_locations_dataframe = pandas.DataFrame(centered_cubic_lattice_points_NOT_contained_in_dils_dict_for_dataframe)
        """

        #live_display.stop()
        # extract the results of the optimization lattice testing for each dil
        #all_dils_optimization_lattices_result_dataframe_list = []
        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
            structureID_dil = specific_dil_structure["ROI"]
            potential_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"]
            zero_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"]

            # Drop the centroid from the lattice! It messes up the contour plot! The centroid was inserted into the first position!
            potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])

            # Extract tested optimization points only in the DIL
            #centered_cubic_lattice_points_contained_only_in_sp_dil_arr = specific_dil_structure["Biopsy optimization: cubic lattice of optimization points only in dil"]

            # Calculate the optimized planes dataframe
            sp_dil_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"]
            guidance_map_max_planes_dataframe = biopsy_optimizer_module_v1_helpers.guidance_map_max_planes_dataframe(potential_optimal_locations_dataframe_centroid_dropped,
                                                                sp_dil_optimal_locations_dataframe,
                                                                voxel_size_for_dil_optimizer_grid,
                                                                zero_locations_dataframe,
                                                                structureID_dil,
                                                                patientUID,
                                                                important_info,
                                                                live_display)

            guidance_map_max_planes_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
                guidance_map_max_planes_dataframe,
                threshold=0.25,
                ignore_types=(np.floating,),
                do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_GUIDANCE_MAP_MAX_PLANES_NEVER_CATEGORICAL_COLUMNS,
            )
            specific_dil_structure["Biopsy optimization: guidance map max-planes dataframe"] = guidance_map_max_planes_dataframe
            

            #all_dils_optimization_lattices_result_dataframe_list.append(potential_optimal_locations_dataframe_centroid_dropped)

        #all_dils_optimization_lattices_result_dataframe = pandas.concat(all_dils_optimization_lattices_result_dataframe_list)
        #del all_dils_optimization_lattices_result_dataframe_list

        # IMPORTANT: Need to investigate this dataframe... The columns should be the same between these two???
        #all_dils_and_non_dil_optimization_lattices_result_dataframe = pandas.concat([all_dils_optimization_lattices_result_dataframe,centered_cubic_lattice_non_dil_locations_dataframe])
        
        #misc_tools.point_remover_from_numpy_arr_v2(all_points_2d_arr, points_to_remove_2d_arr)
        all_zero_locations_dataframe_list = []
        all_potential_optimal_locations_dataframe_centroid_dropped_list = [] # these are the points actually tested (ie the points wihtin the dil!)
        for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
            zero_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"]
            all_zero_locations_dataframe_list.append(zero_locations_dataframe)
                
            potential_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"]
            potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])
            all_potential_optimal_locations_dataframe_centroid_dropped_list.append(potential_optimal_locations_dataframe_centroid_dropped)
        
        all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe = pandas.concat(all_potential_optimal_locations_dataframe_centroid_dropped_list, ignore_index = True)
        
        all_zero_locations_dataframe = misc_tools.intersect_dataframes(all_zero_locations_dataframe_list)
        
        entire_overlapped_lattice_dataframe = pandas.concat([all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe, all_zero_locations_dataframe], ignore_index = True)
        
        #entire_overlapped_lattice_dataframe = biopsy_optimizer.specific_dil_to_all_dils_optimization_lattice_dataframe_combiner(pydicom_item,
                                                            #dil_ref)

        # Calculate the cumulative_projection dataframe
        cumulative_projection_optimization_scores_dataframe = biopsy_optimizer_module_v1_helpers.guidance_map_cumulative_projection_dataframe_creator(entire_overlapped_lattice_dataframe)

        # save the selected prostate to all DILs (they will all contain the same value but its the best way to save this)!
        #for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        #    specific_dil_structure["Biopsy optimization: selected relative prostate dict"] = {"Info": selected_prostate_info, "Centroid vector array": prostate_centroid}
        # changed my mind!


        # save the full intersection zero points
        all_zero_locations_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
            all_zero_locations_dataframe,
            threshold=0.25,
            ignore_types=(np.floating,),
            do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
        )
        pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: All points outside of DILs (zero points) dataframe"] = all_zero_locations_dataframe
        del all_zero_locations_dataframe

        # save the full intersection tested points (ie the points that were actually in the dils)
        all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
            all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe,
            threshold=0.25,
            ignore_types=(np.floating,),
            do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
        )
        pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: All points within DILs (tested points) dataframe"] = all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe
        del all_potential_optimal_locations_dataframe_centroid_dropped_all_dils_dataframe

        # save the full lattice, this will only be useful (i think) for creating the contour plots at the end, ie. doesnt need to be CSVd!!!
        entire_overlapped_lattice_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
            entire_overlapped_lattice_dataframe,
            threshold=0.25,
            ignore_types=(np.floating,),
            do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS,
        )
        pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"] = entire_overlapped_lattice_dataframe
        del entire_overlapped_lattice_dataframe

        # save the cumulative_projection
        cumulative_projection_optimization_scores_dataframe = dataframe_builders.convert_columns_to_categorical_and_downcast(
            cumulative_projection_optimization_scores_dataframe,
            threshold=0.25,
            ignore_types=(np.floating,),
            do_not_convert_column_names_to_categorical=dataframe_dtype_policy.OPTIMIZER_V1_CUMULATIVE_PROJECTION_NEVER_CATEGORICAL_COLUMNS,
        )
        pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Biopsy optimization - Cumulative projection (all points within prostate) dataframe"] = cumulative_projection_optimization_scores_dataframe
        del cumulative_projection_optimization_scores_dataframe

        

        ###
        indeterminate_progress_sub.update(indeterminate_task, visible = False)
        ###

        #live_display.stop()

        
        patients_progress.update(processing_patients_task, advance=1)
        completed_progress.update(processing_patients_task_completed, advance=1)
    patients_progress.update(processing_patients_task, visible=False)
    completed_progress.update(processing_patients_task_completed,  visible=True)



    #live_display.start()

    if display_optimization_contour_plots_bool == True:
        for patientUID,pydicom_item in master_structure_reference_dict.items():

            
            sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]                 

            specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
            selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

            prostate_ID = selected_prostate_info["Structure ID"]
            prostate_ref_type = selected_prostate_info["Struct ref type"]
            prostate_ref_num = selected_prostate_info["Dicom ref num"]
            prostate_structure_index = selected_prostate_info["Index number"]
            prostate_found_bool = selected_prostate_info["Struct found bool"]

            """
            if prostate_found_bool == True:
                selected_prostate_centroid = pydicom_item[prostate_ref_type][prostate_structure_index]["Structure global centroid"].reshape(3)
            else: 
                important_info.add_text_line('Prostate not found! Defaulting prostate centroid to Zero-vector')
                selected_prostate_centroid = np.array([0,0,0])
            """
            #selected_prostate_centroid = pydicom_item["Biopsy optimization: selected relative prostate dict"]["Centroid vector array"]

            optimal_locations_dataframe_list = []
            #dil_centroids_list = []
            dil_centroids_optimization_locations_list = []
            for specific_dil_structure_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
                dil_centroids_optimization_locations_dataframe = specific_dil_structure["Biopsy optimization: DIL centroid optimal biopsy location dataframe"] 
                optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location dataframe"]
                dil_global_centroid = specific_dil_structure["Structure global centroid"]
                
                dil_centroids_optimization_locations_list.append(dil_centroids_optimization_locations_dataframe)
                optimal_locations_dataframe_list.append(optimal_locations_dataframe)
                #dil_centroids_list.append(dil_global_centroid)

            sp_patient_centroid_optimal_dataframe = pandas.concat(dil_centroids_optimization_locations_list)
            sp_patient_optimal_dataframe = pandas.concat(optimal_locations_dataframe_list)

            num_dil_structs_patient_specific = master_structure_info_dict["By patient"][patientUID][dil_ref]["Num structs"]
            
            # changed this in favor of selecting unique structures at the beginning of pipeline and saving to dataframe! See above in the patient loop!
            # can pick a random one, since each dil saved the same information of which prostate was selected for the biopsy optimization
            #random_dil_structure = pydicom_item[dil_ref][random.randint(0,num_dil_structs_patient_specific-1)]
            #selected_prostate_info = random_dil_structure["Biopsy optimization: selected relative prostate dict"]["Info"]
            #selected_prostate_centroid = random_dil_structure["Biopsy optimization: selected relative prostate dict"]["Centroid vector array"]

            entire_overlapped_lattice_dataframe = pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"]

            df_simple = entire_overlapped_lattice_dataframe[['Test location (Prostate centroid origin) (X)','Test location (Prostate centroid origin) (Y)','Test location (Prostate centroid origin) (Z)','Proportion of normal dist points contained']]

            plane_combinations = [(0,1),(0,2),(2,1)] # This defines Transverse (X,Y), Coronal (X,Z) and Saggital (Z,Y)
            for combination in plane_combinations:
                index_to_column_dict = {0: 'Test location (Prostate centroid origin) (X)', 1: 'Test location (Prostate centroid origin) (Y)', 2: 'Test location (Prostate centroid origin) (Z)'}
                dfcumulative = df_simple.groupby([index_to_column_dict[combination[0]],index_to_column_dict[combination[1]]])['Proportion of normal dist points contained'].sum().reset_index()
                max_val = (dfcumulative['Proportion of normal dist points contained']).max()
                dfcumulative['Proportion of normal dist points contained'] = dfcumulative['Proportion of normal dist points contained']/max_val
                
                fig = go.Figure()

                

                for index, row in sp_patient_optimal_dataframe.iterrows():
                    fig.add_scatter(x=[row[dfcumulative.columns[0]]],
                            y=[row[dfcumulative.columns[1]]],
                            marker=dict(
                                color='orange',
                                size=10,
                                symbol = 'circle'
                            ),
                            text=[row["Relative DIL ID"]],
                            mode = "markers+text",
                            name=row["Relative DIL ID"]+' optimal',
                            textposition="bottom center",
                            textfont=dict(
                                family="sans serif",
                                size=12,
                                color="white"
                            )
                        )
                    
                for index, row in sp_patient_centroid_optimal_dataframe.iterrows():
                    fig.add_scatter(x=[row[dfcumulative.columns[0]]],
                            y=[row[dfcumulative.columns[1]]],
                            marker=dict(
                                color='yellow',
                                size=10,
                                symbol = 'circle'
                            ),
                            text=[row["Relative DIL ID"]],
                            mode = "markers+text",
                            name=row["Relative DIL ID"]+' centroid',
                            textposition="bottom center",
                            textfont=dict(
                                family="sans serif",
                                size=12,
                                color="white"
                            )
                        )
                fig.add_scatter(x=[0],
                            y=[0],
                            marker=dict(
                                color='black',
                                size=10,
                                symbol = 'circle'
                            ),
                        name='Prostate centroid')  

                fig.add_trace(
                    go.Contour(
                        z=dfcumulative['Proportion of normal dist points contained'],
                        x=dfcumulative.iloc[:,0],
                        y=dfcumulative.iloc[:,1],
                        colorscale=[[0, 'rgb(0,0,255)'], [0.9, 'rgb(255,0,0)'],[1, 'rgb(0,255,0)']],
                        zmax = 1,
                        zmin = 0,
                        autocontour = False,
                        contours = go.contour.Contours(type = 'levels', showlines = True, coloring = 'heatmap', showlabels = True, size = 0.1),
                        connectgaps = False, 
                        colorbar = go.contour.ColorBar(len = 0.5)
                    ))

                x_axis_name = dfcumulative.columns[0][-2]
                y_axis_name = dfcumulative.columns[1][-2]
                patient_pos_dict = {'X': ' (L/R)', "Y":' (A/P)', "Z": '(S/I)'}
                fig['layout']['xaxis'].update(title=x_axis_name+patient_pos_dict[x_axis_name])
                fig['layout']['yaxis'].update(title=y_axis_name+patient_pos_dict[y_axis_name])
                
                patient_plane_dict = {'XY': ' Transverse (XY)', "YZ": ' Sagittal (YZ)', "XZ": ' Coronal (XZ)',
                                        'YX': ' Transverse (YX)', "ZY": ' Sagittal (ZY)', "ZX": ' Coronal (ZX)'}
                patient_plane_determiner_str = x_axis_name+y_axis_name
                
                fig.add_annotation(text="Cumulative, "+patient_plane_dict[patient_plane_determiner_str]+' plane',
                    xref="paper", yref="paper",
                    x=0.95, y=0.9, showarrow=False,
                        font=dict(family="Courier New, monospace", size=16, color="#ffffff")
                    )  
                fig.add_annotation(text="Patient: "+patientUID,
                                        xref="paper", yref="paper",
                                        x=0.95, y=0.95, showarrow=False,
                                            font=dict(family="Courier New, monospace", size=16, color="#ffffff")
                                        )  
                

                fig.show()   





    ### THIS DATAFRAME IS LARGE, TAKES UP TOO MUCH MEMORY, DELETING AFTER THIS POINT!
    #for patientUID,pydicom_item in master_structure_reference_dict.items():
    #    del pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"]


    return live_display