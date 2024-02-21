import ques_funcs
import sys
import scipy
import numpy as np
import point_containment_tools
import math
import MC_simulator_convex
import cuspatial
import geopandas
from shapely.geometry import Point, Polygon, MultiPoint # for point in polygon test
import plotting_funcs
import cudf
import time

def checkdirs(live_display, important_info, *paths):
    created_a_dir = False
    for path in paths:
        if path.exists():
            important_info.add_text_line(str(path)+ " already exists.", live_display)
        else:
            path.mkdir(parents=True, exist_ok=True)
            important_info.add_text_line("Path "+ str(path)+ " created.", live_display)
            created_a_dir = True
    if created_a_dir == True:
        live_display.stop()
        print('Directories have been created, please ensure the input folder is non-empty, then continue.')
        continue_programme = ques_funcs.ask_ok('> Continue?' )
        if continue_programme == False:
            sys.exit('> Programme exited.')
        else:
            live_display.start()


def find_closest_z_slice(threeD_data_zslice_list,z_val):
    # used to find the closest zslice of points to a given z value within the ThreeDdata structure 
    # which is a list of numpy arrays where each
    # element of the list is a constant zslice
    closest_z_slice_index = min(range(len(threeD_data_zslice_list)), key=lambda i: abs(threeD_data_zslice_list[i][0,2]-z_val))
    return closest_z_slice_index


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [float('NaN')]*(target_len - len(some_list))


def structure_volume_calculator(structure_points_array,
                                interpolated_zvlas_list,
                                zslices_list,
                                structure_info,
                                plot_volume_calculation_containment_result_bool,
                                voxel_size_for_structure_volume_calc,
                                factor_for_voxel_size,
                                cupy_array_upper_limit_NxN_size_input,
                                layout_groups,
                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                progress_bar_level_task_obj,
                                live_display
                                ):
    
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    #with live_display:
    live_display.refresh()
    # two points which are fruthest apart will occur as vertices of the convex hull
    candidates = structure_points_array[scipy.spatial.ConvexHull(structure_points_array).vertices]

    # get distances between each pair of candidate points
    dist_mat = scipy.spatial.distance_matrix(candidates, candidates)

    # get indices of candidates that are furthest apart
    furthest_pt_1, furthest_pt_2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    maximum_distance = np.max(dist_mat)

    lattice_size = math.ceil(maximum_distance*1.15) # must be an integer

    # extract global structure centroid
    # actually not needed
    #structure_global_centroid = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure global centroid"].copy()


    if voxel_size_for_structure_volume_calc == 0:
        voxel_size_for_structure_volume_calc = maximum_distance/factor_for_voxel_size

    interpolated_pts_point_cloud = point_containment_tools.create_point_cloud(structure_points_array)
    interpolated_pts_point_cloud_color = np.array([0,0,1])
    interpolated_pts_point_cloud.paint_uniform_color(interpolated_pts_point_cloud_color)

    axis_aligned_bounding_box = interpolated_pts_point_cloud.get_axis_aligned_bounding_box()
    axis_aligned_bounding_box_points_arr = np.asarray(axis_aligned_bounding_box.get_box_points())
    bounding_box_color = np.array([0,0,0], dtype=float)
    axis_aligned_bounding_box.color = bounding_box_color
    max_bounds = np.amax(axis_aligned_bounding_box_points_arr, axis=0)
    min_bounds = np.amin(axis_aligned_bounding_box_points_arr, axis=0)

    lattice_sizex = int(math.ceil(abs(max_bounds[0]-min_bounds[0])/voxel_size_for_structure_volume_calc) + 1)
    lattice_sizey = int(math.ceil(abs(max_bounds[1]-min_bounds[1])/voxel_size_for_structure_volume_calc) + 1)
    lattice_sizez = int(math.ceil(abs(max_bounds[2]-min_bounds[2])/voxel_size_for_structure_volume_calc) + 1)
    origin = min_bounds

    # generate cubic lattice of points
    centered_cubic_lattice_sp_structure = MC_simulator_convex.generate_cubic_lattice(voxel_size_for_structure_volume_calc, 
                                                                                        lattice_sizex,
                                                                                        lattice_sizey,
                                                                                        lattice_sizez,
                                                                                        origin)
    
    volume_element = voxel_size_for_structure_volume_calc**3
    
    del interpolated_pts_point_cloud

    num_test_points = centered_cubic_lattice_sp_structure.shape[0]

    """
    structure_info = [structureID,
                        structs,
                        structure_reference_number,
                        specific_structure_index
                        ]
    """
    
    # Extract and calculate relative structure info
    #interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
    #interpolated_zvlas_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
    #zslices_list = interslice_interpolation_information.interpolated_pts_list
    max_zval = max(interpolated_zvlas_list)
    min_zval = min(interpolated_zvlas_list)
    zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in zslices_list]
    zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(zslices_polygons_list))

    centered_cubic_lattice_sp_structure_XY = centered_cubic_lattice_sp_structure[:,0:2]
    centered_cubic_lattice_sp_structure_Z = centered_cubic_lattice_sp_structure[:,2]

    #start = time.time()
    #nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_array_input(interpolated_zvlas_list,centered_cubic_lattice_sp_structure_Z)
    #end = time.time()
    #print(end - start)
    
    ### Using cupy is quicker than the old way (above) with the vectorized function (the function was crap anyways)
    ### updated it further with the below generic function!
    #start = time.time()
    #nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.take_closest_cupy(interpolated_zvlas_list, centered_cubic_lattice_sp_structure_Z)
    #end = time.time()
    #print(end - start)
    

    nearest_interpolated_zslice_index_array, nearest_interpolated_zslice_vals_array = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvlas_list, 
                                                                                                                                                            centered_cubic_lattice_sp_structure_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            progress_bar_level_task_obj
                                                                                                                                                            )

    centered_cubic_lattice_sp_structure_XY_interleaved_1darr = centered_cubic_lattice_sp_structure_XY.flatten()
    centered_cubic_lattice_sp_structure_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(centered_cubic_lattice_sp_structure_XY_interleaved_1darr)

    # Test point containment 
    #start = time.time()
    containment_info_grand_pandas_dataframe, live_display = point_containment_tools.cuspatial_points_contained_generic_numpy_pandas(zslices_polygons_cuspatial_geoseries,
        centered_cubic_lattice_sp_structure_XY_cuspatial_geoseries_points, 
        centered_cubic_lattice_sp_structure, 
        nearest_interpolated_zslice_index_array,
        nearest_interpolated_zslice_vals_array,
        max_zval,
        min_zval,
        structure_info,
        layout_groups,
        live_display,
        structures_progress,
        upper_limit_size_input = cupy_array_upper_limit_NxN_size_input
        )
    
    live_display.refresh()
    #end = time.time()
    #print(end - start)               

    if plot_volume_calculation_containment_result_bool == True:
        test_pts_color_R = containment_info_grand_pandas_dataframe["Pt clr R"].to_numpy()
        test_pts_color_G = containment_info_grand_pandas_dataframe["Pt clr G"].to_numpy()
        test_pts_color_B = containment_info_grand_pandas_dataframe["Pt clr B"].to_numpy()
        test_pts_color_arr = np.empty([num_test_points,3])
        test_pts_color_arr[:,0] = test_pts_color_R
        test_pts_color_arr[:,1] = test_pts_color_G
        test_pts_color_arr[:,2] = test_pts_color_B
        centered_cubic_lattice_sp_structure_pcd = point_containment_tools.create_point_cloud_with_colors_array(centered_cubic_lattice_sp_structure, test_pts_color_arr)
        struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(structure_points_array, color = np.array([0,0,1]))
        plotting_funcs.plot_geometries(centered_cubic_lattice_sp_structure_pcd, struct_interpolated_pts_pcd, label='Unknown')                                                                    
        del centered_cubic_lattice_sp_structure_pcd
        del struct_interpolated_pts_pcd

    
    # calculate volume of structure
        
    number_of_contained_points = len(containment_info_grand_pandas_dataframe[containment_info_grand_pandas_dataframe["Pt contained bool"] == True].index)
    structure_volume = number_of_contained_points*volume_element

    del containment_info_grand_pandas_dataframe

    return structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, live_display








def structure_dimensions_calculator(structure_points_array,
                                interpolated_zvlas_list,
                                zslices_list,
                                structure_centroid_vec,
                                structure_info,
                                plot_dimension_calculation_containment_result_bool,
                                voxel_size_for_structure_dimension_calc,
                                factor_for_voxel_size,
                                cupy_array_upper_limit_NxN_size_input,
                                layout_groups,
                                nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                progress_bar_level_task_obj,
                                live_display
                                ):
    app_header,progress_group_info_list,important_info,app_footer = layout_groups
    completed_progress, patients_progress, structures_progress, biopsies_progress, MC_trial_progress, indeterminate_progress_main, indeterminate_progress_sub, progress_group = progress_group_info_list
    
    #with live_display:
    live_display.refresh()
    # two points which are fruthest apart will occur as vertices of the convex hull
    candidates = structure_points_array[scipy.spatial.ConvexHull(structure_points_array).vertices]

    # get distances between each pair of candidate points
    dist_mat = scipy.spatial.distance_matrix(candidates, candidates)

    # get indices of candidates that are furthest apart
    furthest_pt_1, furthest_pt_2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)

    maximum_distance = np.max(dist_mat)

    lattice_size = math.ceil(maximum_distance*1.15) # must be an integer

    # extract global structure centroid
    # actually not needed
    #structure_global_centroid = master_structure_reference_dict[patientUID][structs][specific_structure_index]["Structure global centroid"].copy()


    if voxel_size_for_structure_dimension_calc == 0:
        voxel_size_for_structure_dimension_calc = voxel_size_for_structure_dimension_calc/factor_for_voxel_size

    
    dir_x = np.array([1,0,0])
    dir_y = np.array([0,1,0])
    dir_z = np.array([0,0,1])

    line_of_pts_x = generate_line_of_points(structure_centroid_vec, dir_x, voxel_size_for_structure_dimension_calc, maximum_distance)
    line_of_pts_y = generate_line_of_points(structure_centroid_vec, dir_y, voxel_size_for_structure_dimension_calc, maximum_distance)
    line_of_pts_z = generate_line_of_points(structure_centroid_vec, dir_z, voxel_size_for_structure_dimension_calc, maximum_distance)
    num_pts_in_x_line = line_of_pts_x.shape[0]
    num_pts_in_y_line = line_of_pts_y.shape[0]
    num_pts_in_z_line = line_of_pts_z.shape[0]
    total_num_test_pts = num_pts_in_x_line + num_pts_in_y_line + num_pts_in_z_line

    all_line_of_pts = np.empty((total_num_test_pts,3))
    all_line_of_pts[0:num_pts_in_x_line,:] = line_of_pts_x
    all_line_of_pts[num_pts_in_x_line:num_pts_in_x_line+num_pts_in_y_line,:] = line_of_pts_y
    all_line_of_pts[num_pts_in_x_line+num_pts_in_y_line:,:] = line_of_pts_z

    
    
    # Extract and calculate relative structure info
    max_zval = max(interpolated_zvlas_list)
    min_zval = min(interpolated_zvlas_list)
    zslices_polygons_list = [Polygon(polygon[:,0:2]) for polygon in zslices_list]
    zslices_polygons_cuspatial_geoseries = cuspatial.GeoSeries(geopandas.GeoSeries(zslices_polygons_list))

    

    all_line_of_pts_sp_structure_XY = all_line_of_pts[:,0:2]
    all_line_of_pts_sp_structure_Z = all_line_of_pts[:,2]
    
    
    # Note it is quicker to use the cupy function than the old vectorized function (code above)!
    ### updated further with below generic function
    #start = time.time()
    #nearest_interpolated_zslice_index_array_all, nearest_interpolated_zslice_vals_array_all = point_containment_tools.take_closest_cupy(interpolated_zvlas_list,all_line_of_pts_sp_structure_Z)
    #end = time.time()
    #print(end - start)


    nearest_interpolated_zslice_index_array_all, nearest_interpolated_zslice_vals_array_all = point_containment_tools.nearest_zslice_vals_and_indices_cupy_generic(interpolated_zvlas_list, 
                                                                                                                                                            all_line_of_pts_sp_structure_Z,
                                                                                                                                                            nearest_zslice_vals_and_indices_cupy_generic_max_size,
                                                                                                                                                            progress_bar_level_task_obj
                                                                                                                                                            )  



    line_of_pts_all_sp_structure_XY_interleaved_1darr = all_line_of_pts_sp_structure_XY.flatten()
    line_of_pts_all_sp_structure_XY_cuspatial_geoseries_points = cuspatial.GeoSeries.from_points_xy(line_of_pts_all_sp_structure_XY_interleaved_1darr)

    # Test point containment
    
    ### Quicker to do all the points at once!
    #start = time.time()
    containment_info_grand_pandas_dataframe_all, live_display = point_containment_tools.cuspatial_points_contained_generic_numpy_pandas(zslices_polygons_cuspatial_geoseries,
        line_of_pts_all_sp_structure_XY_cuspatial_geoseries_points, 
        all_line_of_pts, 
        nearest_interpolated_zslice_index_array_all,
        nearest_interpolated_zslice_vals_array_all,
        max_zval,
        min_zval,
        structure_info,
        layout_groups,
        live_display,
        structures_progress,
        upper_limit_size_input = cupy_array_upper_limit_NxN_size_input
        )
    live_display.refresh()

    #end = time.time()
    #print(end - start)  

            

    if plot_dimension_calculation_containment_result_bool == True:
        
        test_pts_color_R = containment_info_grand_pandas_dataframe_all["Pt clr R"].to_numpy()
        test_pts_color_G = containment_info_grand_pandas_dataframe_all["Pt clr G"].to_numpy()
        test_pts_color_B = containment_info_grand_pandas_dataframe_all["Pt clr B"].to_numpy()
        test_pts_color_arr = np.empty([total_num_test_pts,3])
        test_pts_color_arr[:,0] = test_pts_color_R
        test_pts_color_arr[:,1] = test_pts_color_G
        test_pts_color_arr[:,2] = test_pts_color_B

        lines_of_pts = np.vstack((line_of_pts_x,line_of_pts_y,line_of_pts_z))

        lines_of_pts_pcd = point_containment_tools.create_point_cloud_with_colors_array(lines_of_pts, test_pts_color_arr)
        struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(structure_points_array, color = np.array([0,0,1]))
        plotting_funcs.plot_geometries(lines_of_pts_pcd, struct_interpolated_pts_pcd, label='Unknown')                                                                    
        del lines_of_pts_pcd
        del struct_interpolated_pts_pcd
        del containment_info_grand_pandas_dataframe_all


    
    # calculate dimensions of structure
    containment_info_grand_pandas_dataframe_x = containment_info_grand_pandas_dataframe_all.iloc[0:num_pts_in_x_line]
    containment_info_grand_pandas_dataframe_y = containment_info_grand_pandas_dataframe_all.iloc[num_pts_in_x_line:num_pts_in_x_line+num_pts_in_y_line]
    containment_info_grand_pandas_dataframe_z = containment_info_grand_pandas_dataframe_all.iloc[num_pts_in_x_line+num_pts_in_y_line:]
        
    number_of_contained_points_x = len(containment_info_grand_pandas_dataframe_x[containment_info_grand_pandas_dataframe_x["Pt contained bool"] == True].index)
    structure_dimension_length_x = number_of_contained_points_x*voxel_size_for_structure_dimension_calc

    number_of_contained_points_y = len(containment_info_grand_pandas_dataframe_y[containment_info_grand_pandas_dataframe_y["Pt contained bool"] == True].index)
    structure_dimension_length_y = number_of_contained_points_y*voxel_size_for_structure_dimension_calc

    number_of_contained_points_z = len(containment_info_grand_pandas_dataframe_z[containment_info_grand_pandas_dataframe_z["Pt contained bool"] == True].index)
    structure_dimension_length_z = number_of_contained_points_z*voxel_size_for_structure_dimension_calc

    structure_dimension_at_centroid_dict = {"X dimension length at centroid": structure_dimension_length_x,
                                            "Y dimension length at centroid": structure_dimension_length_y,
                                            "Z dimension length at centroid": structure_dimension_length_z
                                            }

    del containment_info_grand_pandas_dataframe_x
    del containment_info_grand_pandas_dataframe_y
    del containment_info_grand_pandas_dataframe_z
    del containment_info_grand_pandas_dataframe_all

    return structure_dimension_at_centroid_dict, voxel_size_for_structure_dimension_calc, live_display






def generate_line_of_points(center, direction_vector, spacing, length):
    """
    Generate points in a line based on the given parameters.

    Parameters:
    - center: The center point of the line (tuple or list of coordinates).
    - direction_vector: The direction vector of the line (tuple or list).
    - spacing: The spacing between consecutive points on the line.
    - length: The length of the line in each direction from the center.

    Returns:
    - A NumPy array of points along the line.
    """
    center = np.array(center)
    direction_vector = np.array(direction_vector)

    # Normalize direction vector
    direction_vector = direction_vector/np.linalg.norm(direction_vector)

    # Calculate points along the line
    points = []
    for i in range(-int(length/spacing), int(length/spacing) + 1):
        point = center + i * spacing * direction_vector
        points.append(point)

    return np.array(points)




def bx_position_classifier_in_prostate_frame_sextant(bx_vec_in_prostate_frame,
                                                     distance_to_midgland_threshold):
    
    # Note that the distance to midgland threshold should be the distance in one direction. 
    # Therefore, it should be a sixth of the total prostate lwength in the z dimension.

    # NOTE THAT distance_to_midgland_threshold SHOULD BE A POSITUVE QUANTITY!

    left_right_pos = None
    sup_inf_pos = None
    ant_post_pos = None

    bx_vec_in_prostate_frame_X = bx_vec_in_prostate_frame[0]
    bx_vec_in_prostate_frame_Y = bx_vec_in_prostate_frame[1]
    bx_vec_in_prostate_frame_Z = bx_vec_in_prostate_frame[2]

    # DETERMINE LEFT/RIGHT
    if bx_vec_in_prostate_frame_X >= 0:
        left_right_pos = 'Left'
    else:
        left_right_pos = 'Right'

    # DETERMINE ANT/POST
    if bx_vec_in_prostate_frame_Y >= 0:
        ant_post_pos = 'Posterior'
    else:
        ant_post_pos = 'Anterior'
    
    # DETERMINE SUP/INF
    if bx_vec_in_prostate_frame_Z >= distance_to_midgland_threshold:
        sup_inf_pos = 'Base (Superior)'
    elif (bx_vec_in_prostate_frame_Z <= distance_to_midgland_threshold) and (bx_vec_in_prostate_frame_Z >= -distance_to_midgland_threshold):
        sup_inf_pos = 'Mid'
    elif bx_vec_in_prostate_frame_Z <= -distance_to_midgland_threshold:
        sup_inf_pos = 'Apex (Inferior)'

    return {"LR":left_right_pos,"AP":ant_post_pos,"SI":sup_inf_pos}




def prostate_selector(pydicom_item,
                      oar_ref,
                      prostate_contour_name):
    
    patient_uid_str = pydicom_item["Patient UID (generated)"]
    prostates_found_list = []
    for specific_oar_structure in pydicom_item[oar_ref]:
        oar_structureID = specific_oar_structure["ROI"]
        if (prostate_contour_name in oar_structureID):

            prostate_structure_info = specific_structure_info_dict_creator('given', specific_structure = specific_oar_structure)
            
            prostates_found_list.append(prostate_structure_info)
            del prostate_structure_info
            

    num_prostates_found = len(prostates_found_list)

    if num_prostates_found >= 1:
        selected_prostate_info = prostates_found_list[0] # select the prostate with the lowest index number 
        prostate_found_bool = True

    else:
        prostate_structure_info_not_found = specific_structure_info_dict_creator('custom',
                                        structid = 'Prostate not found', 
                                        sruct_ref_num = None, 
                                        struct_ref_type = oar_ref, 
                                        struct_index_num = None 
                                        )
        
        prostate_found_bool = False
        
        selected_prostate_info = prostate_structure_info_not_found
        
    if num_prostates_found >= 1:
        message_string = 'Number of prostates found: ' + str(num_prostates_found) + '. Prostate chosen: '+ selected_prostate_info["Structure ID"] + '. Patient: ' + str(patient_uid_str)
    elif num_prostates_found == 0:
        message_string = 'Prostate not found! Patient: ' + str(patient_uid_str)
    return selected_prostate_info, message_string, prostate_found_bool, num_prostates_found



def specific_structure_info_dict_creator(create_type,
                                        specific_structure = None, 
                                        structid = '', 
                                        sruct_ref_num = None, 
                                        struct_ref_type = None, 
                                        struct_index_num = None 
                                        ):
    
    # create_type can be "given", "custom" or "null"


    # creates a dictionary of the information for a particular structure of interest
    if create_type == 'given':
        structureID_entry = specific_structure["ROI"]
        struct_type_entry = specific_structure["Struct type"]
        structure_reference_number_entry = specific_structure["Ref #"]
        structure_index_number_entry = specific_structure["Index number"]

    # creates a blank one
    elif create_type == "null":
        structureID_entry = None 
        struct_type_entry = None
        structure_reference_number_entry = None
        structure_index_number_entry = None
 
    # creates a custom one with the entries given by the user into the function
    elif create_type == 'custom':
        structureID_entry = structid 
        struct_type_entry = struct_ref_type
        structure_reference_number_entry = sruct_ref_num
        structure_index_number_entry = struct_index_num

        

    structure_info = {"Structure ID": structureID_entry,
                    "Struct ref type": struct_type_entry,
                    "Dicom ref num": structure_reference_number_entry,
                    "Index number": structure_index_number_entry
                }
    
    return structure_info