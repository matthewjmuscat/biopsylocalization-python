import scipy
import numpy as np
import math
import point_containment_tools
import MC_simulator_convex
from shapely.geometry import Point, Polygon, MultiPoint # for point in polygon test
import geopandas
import cuspatial
import cupy as cp
import plotting_funcs
import pandas
import cudf
import time
import math_funcs
from itertools import combinations
import misc_tools

def find_dil_optimal_sampling_position(specific_dil_structure,
                                optimal_normal_dist_option,
                                bias_LR_multiplier,
                                bias_AP_multiplier,
                                bias_SI_multiplier,
                                patientUID,
                                structs_referenced_dict,
                                bx_ref,
                                dil_ref,
                                structure_points_array,
                                interpolated_zvlas_list,
                                zslices_list,
                                structure_info,
                                structure_global_centroid,
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
                                progress_bar_level_task_obj,
                                test_lattice_arr = None,
                                all_points_to_set_to_zero_arr = np.array([[]])
                                ):
    

    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, completed_sections_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list


    #with live_display:
    live_display.refresh()

    ###
    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Generating cubic lattice (Size = {} mm)".format(voxel_size_for_dil_optimizer_grid), total = None)
    ###


    # reshape the centroid array 
    structure_global_centroid = structure_global_centroid.reshape((3))

    # create geoseries of the dil structure for containment tests
    max_zval = max(interpolated_zvlas_list)
    min_zval = min(interpolated_zvlas_list)
    zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in zslices_list]
    zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(zslices_polygons_list))
    del zslices_polygons_list


    """
    ### DETERMINE MAXIMUM DISTANCE BETWEEN TWO POINTS OF THE STRUCTURE (DIL)
    # two points which are fruthest apart will occur as vertices of the convex hull
    candidates = structure_points_array[scipy.spatial.ConvexHull(structure_points_array).vertices]

    # get distances between each pair of candidate points
    dist_mat = scipy.spatial.distance_matrix(candidates, candidates)

    # get indices of candidates that are furthest apart
    furthest_pt_1, furthest_pt_2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    maximum_distance = np.max(dist_mat)

    lattice_size = math.ceil(maximum_distance*1.15) # must be an integer

    # dont want sigma to depend on tumor size!!
    #normal_dist_sigma = maximum_distance*normal_dist_sigma_factor_biopsy_optimizer
    """

    if test_lattice_arr is None:
        interpolated_pts_point_cloud = point_containment_tools.create_point_cloud(structure_points_array)
        interpolated_pts_point_cloud_color = np.array([0,0,1])
        interpolated_pts_point_cloud.paint_uniform_color(interpolated_pts_point_cloud_color)

        axis_aligned_bounding_box = interpolated_pts_point_cloud.get_axis_aligned_bounding_box()
        axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
        bounding_box_color = np.array([0,0,0], dtype=float)
        axis_aligned_bounding_box.color = bounding_box_color
        max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
        min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

        lattice_sizex = int(math.ceil(abs(max_bounds[0]-min_bounds[0])/voxel_size_for_dil_optimizer_grid) + 1)
        lattice_sizey = int(math.ceil(abs(max_bounds[1]-min_bounds[1])/voxel_size_for_dil_optimizer_grid) + 1)
        lattice_sizez = int(math.ceil(abs(max_bounds[2]-min_bounds[2])/voxel_size_for_dil_optimizer_grid) + 1)
        origin = min_bounds

        # generate cubic lattice of points
        centered_cubic_lattice_sp_structure = MC_simulator_convex.generate_cubic_lattice(voxel_size_for_dil_optimizer_grid, 
                                                                                            lattice_sizex,
                                                                                            lattice_sizey,
                                                                                            lattice_sizez,
                                                                                            origin)



        centered_cubic_lattice_sp_structure_XY = centered_cubic_lattice_sp_structure[:,0:2]
        centered_cubic_lattice_sp_structure_Z = centered_cubic_lattice_sp_structure[:,2]

        #nearest_interpolated_zslice_for_test_lattice_index_array, nearest_interpolated_zslice_for_test_lattice_vals_array = point_containment_tools.take_closest_cupy(interpolated_zvlas_list, centered_cubic_lattice_sp_structure_Z)

        nearest_interpolated_zslice_for_test_lattice_index_array, nearest_interpolated_zslice_for_test_lattice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvlas_list, 
                                                                                                                                                            centered_cubic_lattice_sp_structure_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            progress_bar_level_task_obj
                                                                                                                                                            )

        centered_cubic_lattice_sp_structure_XY_interleaved_1darr = centered_cubic_lattice_sp_structure_XY.flatten()
        centered_cubic_lattice_sp_structure_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(centered_cubic_lattice_sp_structure_XY_interleaved_1darr)

        # Test point containment to remove points from the potential optimization testing point lattice that are not inside the DIL
        containment_info_for_potential_optimization_points_grand_pandas_dataframe, live_display = point_containment_tools.cuspatial_points_contained_generic_cupy_pandas(zslices_polygons_cuspatial_geoseries,
            centered_cubic_lattice_sp_structure_XY_cuspatial_geoseries_points, 
            centered_cubic_lattice_sp_structure, 
            nearest_interpolated_zslice_for_test_lattice_index_array,
            nearest_interpolated_zslice_for_test_lattice_vals_array,
            max_zval,
            min_zval,
            structure_info,
            layout_groups,
            live_display,
            progress_bar_level_task_obj,
            upper_limit_size_input = cupy_array_upper_limit_NxN_size_input
            )
        live_display.refresh()
        
        containment_info_for_potential_optimization_points_only_internal_points_grand_pandas_dataframe = containment_info_for_potential_optimization_points_grand_pandas_dataframe.drop(containment_info_for_potential_optimization_points_grand_pandas_dataframe[containment_info_for_potential_optimization_points_grand_pandas_dataframe["Pt contained bool"] == False].index).reset_index()


        centered_cubic_lattice_points_contained_only_sp_structure_arr = centered_cubic_lattice_sp_structure[containment_info_for_potential_optimization_points_only_internal_points_grand_pandas_dataframe["index"].to_numpy()]
        #centered_cubic_lattice_sp_structure_with_dil_centroid = np.insert(centered_cubic_lattice_sp_structure, 0, structure_global_centroid, axis=0)

        del containment_info_for_potential_optimization_points_only_internal_points_grand_pandas_dataframe
        del containment_info_for_potential_optimization_points_grand_pandas_dataframe
        del interpolated_pts_point_cloud
        del centered_cubic_lattice_sp_structure
    
    
    else:
        centered_cubic_lattice_points_contained_only_sp_structure_arr = test_lattice_arr


    centered_cubic_lattice_sp_structure_with_dil_centroid = np.insert(centered_cubic_lattice_points_contained_only_sp_structure_arr, 0, structure_global_centroid, axis=0)
    del centered_cubic_lattice_points_contained_only_sp_structure_arr

    num_points_in_cubic_lattice_plus_centroid = centered_cubic_lattice_sp_structure_with_dil_centroid.shape[0]

    if plot_optimization_point_lattice_bool == True:
        # color the dil centroid differently, just for show!
        cubic_lattice_with_centroid_color_arr = np.empty([num_points_in_cubic_lattice_plus_centroid,3])
        # magenta for centroid
        cubic_lattice_with_centroid_color_arr[0,0] = 1
        cubic_lattice_with_centroid_color_arr[0,1] = 0
        cubic_lattice_with_centroid_color_arr[0,2] = 1
        # black for regular grid positions
        cubic_lattice_with_centroid_color_arr[1:,0] = 0
        cubic_lattice_with_centroid_color_arr[1:,1] = 0
        cubic_lattice_with_centroid_color_arr[1:,2] = 0
        
        centered_cubic_lattice_sp_structure_pcd = point_containment_tools.create_point_cloud_with_colors_array(centered_cubic_lattice_sp_structure_with_dil_centroid, cubic_lattice_with_centroid_color_arr)
        struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(structure_points_array, color = np.array([0,0,1]))
        plotting_funcs.plot_geometries(centered_cubic_lattice_sp_structure_pcd, struct_interpolated_pts_pcd, label='Unknown')                                                                    
        del centered_cubic_lattice_sp_structure_pcd
        del struct_interpolated_pts_pcd
    

    


    ###
    indeterminate_progress_sub.update(indeterminate_task, visible = False)
    ###




    ###
    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Generating 3D Normal distribution", total = None)
    ###


    ### GENERATE THE 3D NORMAL DISTRIBUTION
    if optimal_normal_dist_option == 'biopsy_and_dil_sigmas':
        default_biopsy_sigma_x_list = structs_referenced_dict[bx_ref]["Default sigma X"]
        default_biopsy_sigma_y_list = structs_referenced_dict[bx_ref]["Default sigma Y"]
        default_biopsy_sigma_z_list = structs_referenced_dict[bx_ref]["Default sigma Z"]

        default_dil_sigma_x_list = structs_referenced_dict[dil_ref]["Default sigma X"]
        default_dil_sigma_y_list = structs_referenced_dict[dil_ref]["Default sigma Y"]
        default_dil_sigma_z_list = structs_referenced_dict[dil_ref]["Default sigma Z"]

        sigma_x_arr = np.array(default_biopsy_sigma_x_list + default_dil_sigma_x_list)
        sigma_y_arr = np.array(default_biopsy_sigma_y_list + default_dil_sigma_y_list)
        sigma_z_arr = np.array(default_biopsy_sigma_z_list + default_dil_sigma_z_list)
        
        normal_dist_sigma_x = math_funcs.add_in_quadrature(sigma_x_arr)*bias_LR_multiplier/2 
        normal_dist_sigma_y = math_funcs.add_in_quadrature(sigma_y_arr)*bias_AP_multiplier/2  
        normal_dist_sigma_z = math_funcs.add_in_quadrature(sigma_z_arr)*bias_SI_multiplier/2 

    elif optimal_normal_dist_option == 'dil dimension driven':
        sp_dil_features_dataframe = specific_dil_structure["Structure features dataframe"]

        sp_struct_type = structure_info["Struct ref type"]
        sp_struct_refnum = structure_info["Dicom ref num"]
        sp_struct_id = structure_info["Structure ID"]

        sp_dil_features_dataframe_ensured = sp_dil_features_dataframe[(sp_dil_features_dataframe["Patient ID"] == patientUID) & \
                                                                      (sp_dil_features_dataframe["Structure refnum"] == sp_struct_refnum) & \
                                                                      (sp_dil_features_dataframe["Structure type"] == sp_struct_type) & \
                                                                      (sp_dil_features_dataframe["Structure ID"] == sp_struct_id)  ]
        
        sigma_x_arr = np.array([sp_dil_features_dataframe_ensured.at[0,"L/R dimension at centroid"]])
        sigma_y_arr = np.array([sp_dil_features_dataframe_ensured.at[0,"A/P dimension at centroid"]])
        sigma_z_arr = np.array([sp_dil_features_dataframe_ensured.at[0,"S/I dimension at centroid"]])

        normal_dist_sigma_x = math_funcs.add_in_quadrature(sigma_x_arr)*bias_LR_multiplier/2 
        normal_dist_sigma_y = math_funcs.add_in_quadrature(sigma_y_arr)*bias_AP_multiplier/2  
        normal_dist_sigma_z = math_funcs.add_in_quadrature(sigma_z_arr)*bias_SI_multiplier/2 

        
        
        



    three_d_normal_dist_points_x_cupy = cp.random.normal(loc=0,scale=normal_dist_sigma_x, size=num_normal_dist_points_for_biopsy_optimizer)
    three_d_normal_dist_points_y_cupy = cp.random.normal(loc=0,scale=normal_dist_sigma_y, size=num_normal_dist_points_for_biopsy_optimizer)
    three_d_normal_dist_points_z_cupy = cp.random.normal(loc=0,scale=normal_dist_sigma_z, size=num_normal_dist_points_for_biopsy_optimizer)
    three_d_normal_dist_points_cupy = cp.empty((num_normal_dist_points_for_biopsy_optimizer,3))
    three_d_normal_dist_points_cupy[:,0] = three_d_normal_dist_points_x_cupy
    three_d_normal_dist_points_cupy[:,1] = three_d_normal_dist_points_y_cupy
    three_d_normal_dist_points_cupy[:,2] = three_d_normal_dist_points_z_cupy
    three_d_normal_dist_points = cp.asnumpy(three_d_normal_dist_points_cupy)
    num_normal_dist_points = three_d_normal_dist_points.shape[0]


    ###
    indeterminate_progress_sub.update(indeterminate_task, visible = False)
    ###


    ###
    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Prepping for containment", total = None)
    ###

    point_in_cubic_lattice_index_referencer_for_dataframe_arr = np.empty(num_normal_dist_points*num_points_in_cubic_lattice_plus_centroid)
    tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr = np.tile(three_d_normal_dist_points,(num_points_in_cubic_lattice_plus_centroid,1))
    for point_index, point_in_cubic_lattice in enumerate(centered_cubic_lattice_sp_structure_with_dil_centroid):
        tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr[point_index*num_normal_dist_points:point_index*num_normal_dist_points+num_normal_dist_points] = point_in_cubic_lattice + tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr[point_index*num_normal_dist_points:point_index*num_normal_dist_points+num_normal_dist_points]
        point_in_cubic_lattice_index_referencer_for_dataframe_arr[point_index*num_normal_dist_points:point_index*num_normal_dist_points+num_normal_dist_points] = point_index
    # create reference dataframe to join to the results containment dataframe
    index_referencer_dict = {"Optimization lattice point array index": point_in_cubic_lattice_index_referencer_for_dataframe_arr}
    optimization_lattice_index_referencer_dataframe = pandas.DataFrame.from_dict(index_referencer_dict)


    tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY = tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr[:,0:2]
    tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_Z = tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr[:,2]

    #nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input(interpolated_zvlas_list,tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_Z)
    #nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input_cp(interpolated_zvlas_list,tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_Z)

    """
    ### this can take some time for many points!
    nearest_interpolated_zslice_index_array = np.empty(num_points_in_cubic_lattice_plus_centroid*num_normal_dist_points, dtype = np.int64)
    nearest_interpolated_zslice_vals_array = np.empty(num_points_in_cubic_lattice_plus_centroid*num_normal_dist_points)
    for optimization_point_index in range(num_points_in_cubic_lattice_plus_centroid):
        ### Note cant do all the points at once on the gpu as it is too memory intensive, but breaking it up and performing it on the gpu is 
        ### still faster than what was done above with the old function! Doing it in the same way as below but with numpy is slower and very cpu memory intensive to do it without 
        ### breaking it up
        nearest_interpolated_zslice_index_array_intermediate, nearest_interpolated_zslice_vals_intermediate = point_containment_tools.take_closest_cupy(interpolated_zvlas_list, tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_Z[optimization_point_index*num_normal_dist_points:optimization_point_index*num_normal_dist_points + num_normal_dist_points])
        nearest_interpolated_zslice_index_array[optimization_point_index*num_normal_dist_points:optimization_point_index*num_normal_dist_points + num_normal_dist_points] = nearest_interpolated_zslice_index_array_intermediate
        nearest_interpolated_zslice_vals_array[optimization_point_index*num_normal_dist_points:optimization_point_index*num_normal_dist_points + num_normal_dist_points] = nearest_interpolated_zslice_vals_intermediate
    ###
    """  

    ### CUPY IS ABOUT 10X FASTER!
    """
    st = time.time()
    nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_numpy_generic(interpolated_zvlas_list, 
                                                                                                                                                            tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_numpy_generic_max_size,
                                                                                                                                                            progress_bar_level_task_obj
                                                                                                                                                            )
    et = time.time()
    elapsed_time = et - st
    print('\n Execution time (numpy):', elapsed_time, 'seconds')
    """
    nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvlas_list, 
                                                                                                                                                            tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            progress_bar_level_task_obj
                                                                                                                                                            )

    tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY_interleaved_1darr = tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY.flatten()
    tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY_interleaved_1darr)



    ###
    indeterminate_progress_sub.update(indeterminate_task, visible = False)
    ###



    ###
    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Performing containment of all distributions", total = None)
    ###


    # Test point containment 
    containment_info_grand_pandas_dataframe, live_display = point_containment_tools.cuspatial_points_contained_generic_cupy_pandas(zslices_polygons_cuspatial_geoseries,
        tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY_cuspatial_geoseries_points, 
        tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr, 
        nearest_interpolated_zslice_index_array,
        nearest_interpolated_zslice_vals_array,
        max_zval,
        min_zval,
        structure_info,
        layout_groups,
        live_display,
        progress_bar_level_task_obj,
        upper_limit_size_input = cupy_array_upper_limit_NxN_size_input
        )
    live_display.refresh()


    del tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY_interleaved_1darr
    del tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr_XY_cuspatial_geoseries_points
    

    ###
    indeterminate_progress_sub.update(indeterminate_task, visible = False)
    ###



    ###
    indeterminate_task = indeterminate_progress_sub.add_task("[cyan]~~Tying results in a bow", total = None)
    ###
    

    # this below code does not produce the correct ordering!
    #containment_info_grand_cudf_dataframe_with_optimization_point_array_index_reference = optimization_lattice_index_referencer_dataframe.join(containment_info_grand_cudf_dataframe)
    
    # USE THIS INSTEAD!
    containment_info_grand_cudf_dataframe_with_optimization_point_array_index_reference = pandas.concat([optimization_lattice_index_referencer_dataframe,containment_info_grand_pandas_dataframe],axis=1)
    del containment_info_grand_pandas_dataframe
    del optimization_lattice_index_referencer_dataframe


    # NOTE THAT THE POINT_IN_CUBIC_LATTICE ARE THE POTENTIAL OPTIMAL POSITIONS BEING TESTED!
    test_location_to_dil_centroid_arr = structure_global_centroid - centered_cubic_lattice_sp_structure_with_dil_centroid
    distance_to_dil_centroid_arr = np.linalg.norm(test_location_to_dil_centroid_arr, axis=1) 
    prostate_centroid_to_test_location_arr = centered_cubic_lattice_sp_structure_with_dil_centroid - prostate_centroid
    distance_to_prostate_centroid_arr = np.linalg.norm(prostate_centroid_to_test_location_arr, axis=1) 
    number_of_contained_points_arr = containment_info_grand_cudf_dataframe_with_optimization_point_array_index_reference.groupby("Optimization lattice point array index",sort=True)['Pt contained bool'].sum().to_numpy()
    proportion_of_contained_points_arr = number_of_contained_points_arr/num_normal_dist_points_for_biopsy_optimizer

        
    potential_optimal_locations_dict_for_dataframe = {"Patient ID": [patientUID]*num_points_in_cubic_lattice_plus_centroid,
                                                    "Relative DIL ID": [structure_info["Structure ID"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    "Relative DIL type": [structure_info["Struct ref type"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    "Relative DIL ref num": [structure_info["Dicom ref num"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    "Relative DIL index": [structure_info["Index number"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    #'Test location vector': [tuple(vec) for vec in centered_cubic_lattice_sp_structure_with_dil_centroid],
                                                    'Test location (X)': centered_cubic_lattice_sp_structure_with_dil_centroid[:,0],
                                                    'Test location (Y)': centered_cubic_lattice_sp_structure_with_dil_centroid[:,1],
                                                    'Test location (Z)': centered_cubic_lattice_sp_structure_with_dil_centroid[:,2],
                                                    #'Test location to DIL centroid vector': [tuple(vec) for vec in test_location_to_dil_centroid_arr],
                                                    'Test location to DIL centroid (X)': test_location_to_dil_centroid_arr[:,0],
                                                    'Test location to DIL centroid (Y)': test_location_to_dil_centroid_arr[:,1],
                                                    'Test location to DIL centroid (Z)': test_location_to_dil_centroid_arr[:,2],
                                                    'Dist to DIL centroid': distance_to_dil_centroid_arr,
                                                    'Selected prostate ROI': [selected_prostate_info["Structure ID"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    'Selected prostate type': [selected_prostate_info["Struct ref type"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    'Selected prostate ref num': [selected_prostate_info["Dicom ref num"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    'Selected prostate index': [selected_prostate_info["Index number"]]*num_points_in_cubic_lattice_plus_centroid,
                                                    #'Test location vector (Prostate centroid origin)': [tuple(vec) for vec in prostate_centroid_to_test_location_arr],
                                                    'Test location (Prostate centroid origin) (X)': prostate_centroid_to_test_location_arr[:,0],
                                                    'Test location (Prostate centroid origin) (Y)': prostate_centroid_to_test_location_arr[:,1],
                                                    'Test location (Prostate centroid origin) (Z)': prostate_centroid_to_test_location_arr[:,2],
                                                    'Dist to Prostate centroid': distance_to_prostate_centroid_arr,
                                                    'Number of normal dist points contained': number_of_contained_points_arr,
                                                    'Number of normal dist points tested': [num_normal_dist_points_for_biopsy_optimizer]*num_points_in_cubic_lattice_plus_centroid,
                                                    'Proportion of normal dist points contained': proportion_of_contained_points_arr,
                                                    'Pt actually tested bool': [True]*num_points_in_cubic_lattice_plus_centroid
                                                    }
    
    potential_optimal_locations_dataframe = pandas.DataFrame(potential_optimal_locations_dict_for_dataframe)
    del potential_optimal_locations_dict_for_dataframe
    
    # SHOW THE RESULTS OF THE CONTAINMENT TEST FOR EACH NORMAL DIST FOR EACH OPTIMIZATION LATTICE POINT?               
    if plot_each_normal_dist_containment_result_bool == True:
        for point_index, point_in_cubic_lattice in enumerate(centered_cubic_lattice_sp_structure_with_dil_centroid):
    
            # Extract the relevant tested points containment data 
            containment_info_grand_cudf_dataframe_sp_optimal_test_point = containment_info_grand_cudf_dataframe_with_optimization_point_array_index_reference[containment_info_grand_cudf_dataframe_with_optimization_point_array_index_reference["Optimization lattice point array index"] == point_index]
            threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr = tiled_threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr[point_index*num_normal_dist_points:point_index*num_normal_dist_points+num_normal_dist_points]

            test_pts_color_R = containment_info_grand_cudf_dataframe_sp_optimal_test_point["Pt clr R"].to_numpy()
            test_pts_color_G = containment_info_grand_cudf_dataframe_sp_optimal_test_point["Pt clr G"].to_numpy()
            test_pts_color_B = containment_info_grand_cudf_dataframe_sp_optimal_test_point["Pt clr B"].to_numpy()
            test_pts_color_arr = np.empty([num_normal_dist_points,3])
            test_pts_color_arr[:,0] = test_pts_color_R
            test_pts_color_arr[:,1] = test_pts_color_G
            test_pts_color_arr[:,2] = test_pts_color_B
            normal_dist_centered_on_cubic_lattice_pt_sp_structure_pcd = point_containment_tools.create_point_cloud_with_colors_array(threeD_normal_dist_centered_at_sp_cubic_lattice_point_arr, test_pts_color_arr)
            struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(structure_points_array, color = np.array([0,0,1]))
            plotting_funcs.plot_geometries(normal_dist_centered_on_cubic_lattice_pt_sp_structure_pcd, struct_interpolated_pts_pcd, label='Unknown')                                                                    
            del normal_dist_centered_on_cubic_lattice_pt_sp_structure_pcd
            del struct_interpolated_pts_pcd

        
    # PICK OUT THE OPTIMAL POSITION BY SELECTING THE HIGHEST PROPORTION OF POINTS CONTAINED! (If more than one, select one closest to DIL, if the optimal points are all equidistant to dil, then randomly select one)
    #live_display.stop()
    optimal_locations_dataframe = potential_optimal_locations_dataframe[potential_optimal_locations_dataframe['Number of normal dist points contained'] == potential_optimal_locations_dataframe['Number of normal dist points contained'].max()].reset_index(drop = True)
    num_optimal_points = len(optimal_locations_dataframe)
    #live_display.start()
    if num_optimal_points > 1:
        # If there are ties for the optimal position, pick the one closest to the dil centroid! 
        optimal_locations_dataframe = optimal_locations_dataframe[optimal_locations_dataframe['Dist to DIL centroid'] == optimal_locations_dataframe['Dist to DIL centroid'].min()]
        num_optimal_points = len(optimal_locations_dataframe)
        if num_optimal_points > 1:
            # If they are still tied, pick random one
            optimal_locations_dataframe = optimal_locations_dataframe.sample()
        
        optimal_locations_dataframe = optimal_locations_dataframe.reset_index(drop=True)
    

    # SHOW THE OPTIMIZED LOCATION?
    if show_optimization_point_bool == True:
        struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(structure_points_array, color = np.array([0,0,1]))
        optimal_points_np_array = np.empty([num_optimal_points,3])
        for arr_index, (index, row) in enumerate(optimal_locations_dataframe.iterrows()):
            optimal_points_np_array[arr_index,0:3] = np.array([row['Test location (X)'], row['Test location (Y)'], row['Test location (Z)']])
        # optimal positions are colored green
        optimal_points_pcd = point_containment_tools.create_point_cloud(optimal_points_np_array.reshape((-1,3)), color = np.array([0,1,0]))
        structure_global_centroid_pcd = point_containment_tools.create_point_cloud(structure_global_centroid.reshape((-1,3)), color = np.array([1,0,1])) 
        plotting_funcs.plot_geometries(optimal_points_pcd, struct_interpolated_pts_pcd,structure_global_centroid_pcd, label='Unknown')                                                                    



    ###
    indeterminate_progress_sub.update(indeterminate_task, visible = False)
    ###

    # NOTE THAT THE POINT_IN_CUBIC_LATTICE ARE THE POTENTIAL OPTIMAL POSITIONS BEING TESTED!
    #test_location_to_dil_centroid_arr = structure_global_centroid - centered_cubic_lattice_sp_structure_with_dil_centroid
    #distance_to_dil_centroid_arr = np.linalg.norm(test_location_to_dil_centroid_arr, axis=1) 
    #prostate_centroid_to_test_location_arr = centered_cubic_lattice_sp_structure_with_dil_centroid - prostate_centroid
    #distance_to_prostate_centroid_arr = np.linalg.norm(prostate_centroid_to_test_location_arr, axis=1) 
    #number_of_contained_points_arr = containment_info_grand_cudf_dataframe_with_optimization_point_array_index_reference.groupby("Optimization lattice point array index",sort=True)['Pt contained bool'].sum().to_numpy()
    #proportion_of_contained_points_arr = number_of_contained_points_arr/num_normal_dist_points_for_biopsy_optimizer

    num_points_in_zero_cubic_lattice = all_points_to_set_to_zero_arr.shape[0]
    zero_location_to_dil_centroid_arr = structure_global_centroid - all_points_to_set_to_zero_arr
    zero_lattice_distance_to_dil_centroid_arr = np.linalg.norm(zero_location_to_dil_centroid_arr, axis=1) 
    prostate_centroid_to_zero_location_arr = all_points_to_set_to_zero_arr - prostate_centroid
    zero_lattice_distance_to_prostate_centroid_arr = np.linalg.norm(prostate_centroid_to_zero_location_arr, axis=1) 

    zero_locations_dict_for_dataframe = {"Patient ID": [patientUID]*num_points_in_zero_cubic_lattice,
                                                    "Relative DIL ID": [structure_info["Structure ID"]]*num_points_in_zero_cubic_lattice,
                                                    "Relative DIL type": [structure_info["Struct ref type"]]*num_points_in_zero_cubic_lattice,
                                                    "Relative DIL ref num": [structure_info["Dicom ref num"]]*num_points_in_zero_cubic_lattice,
                                                    "Relative DIL index": [structure_info["Index number"]]*num_points_in_zero_cubic_lattice,
                                                    #'Test location vector': [tuple(vec) for vec in all_points_to_set_to_zero_arr],
                                                    'Test location (X)': all_points_to_set_to_zero_arr[:,0],
                                                    'Test location (Y)': all_points_to_set_to_zero_arr[:,1],
                                                    'Test location (Z)': all_points_to_set_to_zero_arr[:,2],
                                                    #'Test location to DIL centroid vector': [tuple(vec) for vec in zero_location_to_dil_centroid_arr],
                                                    'Test location to DIL centroid (X)': zero_location_to_dil_centroid_arr[:,0],
                                                    'Test location to DIL centroid (Y)': zero_location_to_dil_centroid_arr[:,1],
                                                    'Test location to DIL centroid (Z)': zero_location_to_dil_centroid_arr[:,2],
                                                    'Dist to DIL centroid': zero_lattice_distance_to_dil_centroid_arr,
                                                    'Selected prostate ROI': [selected_prostate_info["Structure ID"]]*num_points_in_zero_cubic_lattice,
                                                    'Selected prostate type': [selected_prostate_info["Struct ref type"]]*num_points_in_zero_cubic_lattice,
                                                    'Selected prostate ref num': [selected_prostate_info["Dicom ref num"]]*num_points_in_zero_cubic_lattice,
                                                    'Selected prostate index': [selected_prostate_info["Index number"]]*num_points_in_zero_cubic_lattice,
                                                    #'Test location vector (Prostate centroid origin)': [tuple(vec) for vec in prostate_centroid_to_zero_location_arr],
                                                    'Test location (Prostate centroid origin) (X)': prostate_centroid_to_zero_location_arr[:,0],
                                                    'Test location (Prostate centroid origin) (Y)': prostate_centroid_to_zero_location_arr[:,1],
                                                    'Test location (Prostate centroid origin) (Z)': prostate_centroid_to_zero_location_arr[:,2],
                                                    'Dist to Prostate centroid': zero_lattice_distance_to_prostate_centroid_arr,
                                                    'Number of normal dist points contained': [0]*num_points_in_zero_cubic_lattice,
                                                    'Number of normal dist points tested': [0]*num_points_in_zero_cubic_lattice,
                                                    'Proportion of normal dist points contained': [0]*num_points_in_zero_cubic_lattice,
                                                    'Pt actually tested bool': [False]*num_points_in_zero_cubic_lattice
                                                    }
    
    zero_locations_dataframe = pandas.DataFrame(zero_locations_dict_for_dataframe)
    del zero_locations_dict_for_dataframe


    return optimal_locations_dataframe, potential_optimal_locations_dataframe, zero_locations_dataframe, live_display





def guidance_map_cumulative_projection_dataframe_creator(all_dils_and_non_dil_optimization_lattices_result_dataframe):
    
    #all_dils_and_non_dil_optimization_lattices_result_dataframe = pydicom_item[all_ref_key]["Multi-structure information dict (not for csv output)"]["Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe"]

    df_simple = all_dils_and_non_dil_optimization_lattices_result_dataframe[['Test location (Prostate centroid origin) (X)','Test location (Prostate centroid origin) (Y)','Test location (Prostate centroid origin) (Z)','Proportion of normal dist points contained']]
    
    dfcumulative_all_planes = pandas.DataFrame()
    coord_index_arr = np.array([0,1,2])
    plane_combinations = [(0,1),(0,2),(2,1)] # This defines Transverse (X,Y), Coronal (X,Z) and Saggital (Z,Y)

    for combination in plane_combinations:
        index_to_column_dict = {0: 'Test location (Prostate centroid origin) (X)', 
                                1: 'Test location (Prostate centroid origin) (Y)', 
                                2: 'Test location (Prostate centroid origin) (Z)'
                                }
        
        dfcumulative = df_simple.groupby([index_to_column_dict[combination[0]],index_to_column_dict[combination[1]]])['Proportion of normal dist points contained'].sum().reset_index()
        max_val = (dfcumulative['Proportion of normal dist points contained']).max()
        dfcumulative['Proportion of normal dist points contained'] = dfcumulative['Proportion of normal dist points contained']/max_val

        x_axis_name = dfcumulative.columns[0][-2]
        y_axis_name = dfcumulative.columns[1][-2]

        patient_plane_dict = {'XY': ' Transverse (XY)', "YZ": ' Sagittal (YZ)', "XZ": ' Coronal (XZ)',
                              'YX': ' Transverse (YX)', "ZY": ' Sagittal (ZY)', "ZX": ' Coronal (ZX)'}
        patient_plane_determiner_str = x_axis_name+y_axis_name

        dfcumulative['Patient plane'] = patient_plane_dict[patient_plane_determiner_str]
        dfcumulative['Coord 1 name'] = index_to_column_dict[combination[0]]
        dfcumulative['Coord 2 name'] = index_to_column_dict[combination[1]]

        dfcumulative.rename(columns={index_to_column_dict[combination[0]]: "Coordinate 1", index_to_column_dict[combination[1]]: "Coordinate 2"}, inplace = True)

        dfcumulative_all_planes = pandas.concat([dfcumulative_all_planes, dfcumulative])


    return dfcumulative_all_planes



def guidance_map_max_planes_dataframe(sp_dil_potential_optimal_locations_dataframe_centroid_dropped,
                                      sp_dil_optimal_locations_dataframe,
                                      voxel_size_for_dil_optimizer_grid,
                                      zero_locations_dataframe,
                                      structureID_dil,
                                      patientUID,
                                      important_info,
                                      live_display):
    
    # swapped out this line because getting rid of the vector columns in the dataframes, they are redundant and take up a lot of memory!
    #sp_dil_optimal_location_vec = np.array(sp_dil_optimal_locations_dataframe.iloc[0]['Test location vector (Prostate centroid origin)'])
    # sp_dil_optimal_location_vec = np.array([sp_dil_optimal_locations_dataframe.at[0,'Test location (Prostate centroid origin) (X)'], 
    #                                         sp_dil_optimal_locations_dataframe.at[0,'Test location (Prostate centroid origin) (Y)'], 
    #                                         sp_dil_optimal_locations_dataframe.at[0,'Test location (Prostate centroid origin) (Z)']])

    

    #point_containment_tools.take_closest_numpy(indexed_array, np.array([sp_dil_optimal_location_zval]))

    df_simple = sp_dil_potential_optimal_locations_dataframe_centroid_dropped[['Test location (Prostate centroid origin) (X)',
                                                                               'Test location (Prostate centroid origin) (Y)',
                                                                               'Test location (Prostate centroid origin) (Z)',
                                                                               'Proportion of normal dist points contained',
                                                                               'X_plane_index',
                                                                               'Y_plane_index',
                                                                               'Z_plane_index']]
    

    
    zero_locations_simple_dataframe = zero_locations_dataframe[['Test location (Prostate centroid origin) (X)',
                                                                               'Test location (Prostate centroid origin) (Y)',
                                                                               'Test location (Prostate centroid origin) (Z)',
                                                                               'Proportion of normal dist points contained',
                                                                               'X_plane_index',
                                                                               'Y_plane_index',
                                                                               'Z_plane_index']]

    index_to_column_dict = {0: 'Test location (Prostate centroid origin) (X)', 
                            1: 'Test location (Prostate centroid origin) (Y)', 
                            2: 'Test location (Prostate centroid origin) (Z)'
                        }
    
    index_to_coord_dict = {0: 'X', 
                           1: 'Y', 
                           2: 'Z'
                        }
    
    dfmax_all_planes = pandas.DataFrame()
    coord_index_arr = np.array([0,1,2])
    plane_combinations = [(0,1),(0,2),(2,1)] # This defines Transverse (X,Y), Coronal (X,Z) and Saggital (Z,Y)
    for combination in plane_combinations:
        
        
        # this assumes there are only 3 coordindates.. we live in 3 dimensions after all! ;)
        const_plane_coordinate = np.setdiff1d(coord_index_arr, combination)[0] # setdiff1d returns the values that are in arr1 that are not in arr2
        
        const_plane_coordinate_str = index_to_coord_dict[const_plane_coordinate]

        
        #sp_dil_optimal_location_sp_coord_val = sp_dil_optimal_location_vec[const_plane_coordinate]
        
        const_plane_index = sp_dil_optimal_locations_dataframe.at[0,f'{const_plane_coordinate_str}_plane_index']

        # select the points from the original all potential optimization points dataframe within the dil that are within a certain tolerance of the max plane coordinate
        # sp_dil_max_plane_df = df_simple[(df_simple[index_to_column_dict[const_plane_coordinate]] <= sp_dil_optimal_location_sp_coord_val + 0.25*voxel_size_for_dil_optimizer_grid) &
        #                                  (df_simple[index_to_column_dict[const_plane_coordinate]] >= sp_dil_optimal_location_sp_coord_val - 0.25*voxel_size_for_dil_optimizer_grid)]#[[index_to_column_dict[combination[0]], index_to_column_dict[combination[1]], 'Proportion of normal dist points contained']]
        
        # modified to allow for collecting constant plane by adding a plane index column earlier in the code!
        sp_dil_max_plane_df = df_simple[df_simple[f'{const_plane_coordinate_str}_plane_index'] == const_plane_index]
        
        # Check if there are duplicate points in the plane
        if sp_dil_max_plane_df.duplicated(subset=[index_to_column_dict[combination[0]],index_to_column_dict[combination[1]]]).any() == True:
            message_string = f"PatientID: {patientUID}, Bx ID: {structureID_dil} | Duplicates found in your max-plane optimization dataframe! Your lattice may be on an angle relative to the planes of interest!"
            important_info.add_text_line(message_string,live_display)
            # Keep only the first lattice point of the duplicates, this is a quick and dirty fix! 
            sp_dil_max_plane_df = sp_dil_max_plane_df[sp_dil_max_plane_df.duplicated(subset=[index_to_column_dict[combination[0]],index_to_column_dict[combination[1]]], keep = 'first') == False]

        # Removed max_val normalization
        #max_val = (sp_dil_max_plane_df['Proportion of normal dist points contained']).max()
        #sp_dil_max_plane_df['Proportion of normal dist points contained'] = dfcumulative['Proportion of normal dist points contained']/max_val

        
        # Extract points in the max plane of interest that are not in the dil
        # points_not_in_sp_dil_max_plane_simple_df = zero_locations_simple_dataframe[(zero_locations_simple_dataframe[index_to_column_dict[const_plane_coordinate]] <= sp_dil_optimal_location_sp_coord_val + 0.25*voxel_size_for_dil_optimizer_grid) &
        #                                  (zero_locations_simple_dataframe[index_to_column_dict[const_plane_coordinate]] >= sp_dil_optimal_location_sp_coord_val - 0.25*voxel_size_for_dil_optimizer_grid)]#[[index_to_column_dict[combination[0]], index_to_column_dict[combination[1]], 'Proportion of normal dist points contained']]
        
        
        # modified to allow for collecting constant plane by adding a plane index column earlier in the code!
        points_not_in_sp_dil_max_plane_simple_df = zero_locations_simple_dataframe[zero_locations_simple_dataframe[f'{const_plane_coordinate_str}_plane_index'] == const_plane_index]

        entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df = pandas.concat([sp_dil_max_plane_df,points_not_in_sp_dil_max_plane_simple_df])
 
        hor_axis_name = index_to_column_dict[combination[0]][-2]
        vert_axis_name = index_to_column_dict[combination[1]][-2]

        patient_plane_dict = {'XY': ' Transverse (XY)', "YZ": ' Sagittal (YZ)', "XZ": ' Coronal (XZ)',
                              'YX': ' Transverse (YX)', "ZY": ' Sagittal (ZY)', "ZX": ' Coronal (ZX)'}
        patient_plane_determiner_str = hor_axis_name+vert_axis_name

        entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df['Patient plane'] = patient_plane_dict[patient_plane_determiner_str]
        entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df['Coord 1 name'] = index_to_column_dict[combination[0]]
        entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df['Coord 2 name'] = index_to_column_dict[combination[1]]
        entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df['Const coord name'] = index_to_column_dict[const_plane_coordinate]

        #entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df.rename(index={0: "Coordinate 1", 1: "Coordinate 2"}, inplace = True)

        dfmax_all_planes = pandas.concat([dfmax_all_planes, entire_lattice_sp_dil_with_other_dils_set_to_0_simple_df])


    return dfmax_all_planes




def specific_dil_to_all_dils_optimization_lattice_dataframe_combiner(pydicom_item,
                                                                     dil_ref):
    
    entire_overlapped_lattice_dataframe = pandas.DataFrame()
    for specific_dil_structure in pydicom_item[dil_ref]: 
        potential_optimal_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (all tested lattice points) dataframe"]
        potential_optimal_locations_dataframe_centroid_dropped = potential_optimal_locations_dataframe.drop([0])
        zero_locations_dataframe = specific_dil_structure["Biopsy optimization: Optimal biopsy location (zero lattice) dataframe"]

        entire_overlapped_lattice_dataframe = pandas.concat([entire_overlapped_lattice_dataframe,potential_optimal_locations_dataframe_centroid_dropped,zero_locations_dataframe]).reset_index(drop= True)

    return entire_overlapped_lattice_dataframe
