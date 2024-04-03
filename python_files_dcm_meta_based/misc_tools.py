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
import open3d as o3d
import misc_tools
from scipy import stats
import meshing_tools
from sklearn.decomposition import PCA


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
                                plot_binary_mask_bool,
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

    ## get binary mask
        
    binary_mask_dataframe = containment_info_grand_pandas_dataframe[containment_info_grand_pandas_dataframe["Pt contained bool"] == True]
    binary_mask_x_vals_arr = binary_mask_dataframe["Test pt X"].to_numpy()
    binary_mask_y_vals_arr = binary_mask_dataframe["Test pt Y"].to_numpy()
    binary_mask_z_vals_arr = binary_mask_dataframe["Test pt Z"].to_numpy()
    binary_mask_arr = np.column_stack(((binary_mask_x_vals_arr, binary_mask_y_vals_arr,binary_mask_z_vals_arr)))

    if plot_binary_mask_bool == True:
        binary_mask_color_R = binary_mask_dataframe["Pt clr R"].to_numpy()
        binary_mask_color_G = binary_mask_dataframe["Pt clr G"].to_numpy()
        binary_mask_color_B = binary_mask_dataframe["Pt clr B"].to_numpy()
        binary_mask_color_arr = np.empty([number_of_contained_points,3])
        binary_mask_color_arr[:,0] = binary_mask_color_R
        binary_mask_color_arr[:,1] = binary_mask_color_G
        binary_mask_color_arr[:,2] = binary_mask_color_B
        binary_mask_sp_structure_pcd = point_containment_tools.create_point_cloud_with_colors_array(binary_mask_arr, binary_mask_color_arr)
        struct_interpolated_pts_pcd = point_containment_tools.create_point_cloud(structure_points_array, color = np.array([0,0,1]))
        plotting_funcs.plot_geometries(binary_mask_sp_structure_pcd, struct_interpolated_pts_pcd, label='Unknown')                                                                    
        del binary_mask_sp_structure_pcd
        del struct_interpolated_pts_pcd
        


    del containment_info_grand_pandas_dataframe

    return structure_volume, maximum_distance, voxel_size_for_structure_volume_calc, binary_mask_arr, live_display








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





def delete_shared_rows(arr1, arr2):
    """
    Deletes rows in arr1 that share rows with arr2.
    
    Parameters:
    arr1 (numpy.ndarray): The first array.
    arr2 (numpy.ndarray): The second array.
    
    Returns:
    numpy.ndarray: The modified arr1 with shared rows removed.
    """
    # Convert arrays to sets of tuples for efficient comparison
    set1 = set(map(tuple, arr1))
    set2 = set(map(tuple, arr2))
    
    # Find the rows in arr1 that are not in arr2
    result = np.array([row for row in arr1 if tuple(row) not in set2])
    
    return result






def compute_curvature(point_cloud, radius):
    # Create KDTree for fast nearest neighbor search
    kdtree = o3d.geometry.KDTreeFlann(point_cloud)

    curvatures = []
    for i in range(len(point_cloud.points)):
        # Query neighbors within a specified radius
        [k, idx, _] = kdtree.search_radius_vector_3d(point_cloud.points[i], radius)
        if k < 3:
            [_, idx, _] = kdtree.search_knn_vector_3d(point_cloud.points[i], 3)

        # Extract neighbors
        neighbors = point_cloud.select_by_index(idx)

        # Fit plane to neighbors using PCA
        _, _, V = np.linalg.svd(np.asarray(neighbors.points) - np.mean(np.asarray(neighbors.points),axis=0))
        normal = V[2]  # Normal to the local plane

        # Compute curvature as 1 - dot product of point normal and plane normal
        curvature = 1 - np.abs(np.dot(neighbors.normals, normal))  # Range: [0, 1]
        curvatures.append(curvature.mean())

    return curvatures



def determine_structure_curvature_dictionary_output(threeDdata_array_fully_interpolated_with_end_caps,
                                                   radius_for_normals_estimation,
                                                   max_nn_for_normals_estimation,
                                                   radius_for_curvature_estimation,
                                                   display_curvature_bool):
    
    point_cloud_fully_interp_with_end_caps = point_containment_tools.create_point_cloud(threeDdata_array_fully_interpolated_with_end_caps)
    
    # estimate point normals
    point_cloud_fully_interp_with_end_caps.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_for_normals_estimation, max_nn=max_nn_for_normals_estimation))
    
    # Compute curvature with a specified radius for local neighborhood
    curvature_values_by_numpy_arr_index_fully_interp_with_end_caps = misc_tools.compute_curvature(point_cloud_fully_interp_with_end_caps, radius = radius_for_curvature_estimation)

    curvature_dict = {"Curvature distribution by numpy index of fully interpolated array with end caps": curvature_values_by_numpy_arr_index_fully_interp_with_end_caps,
                        "Mean curvature": np.mean(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps),
                        "Standard deviation curvature": np.std(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps),
                        "Standard error in mean curvature": stats.sem(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps),
                        "Max curvature": np.max(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps),
                        "Min curvature": np.min(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps)}
    


    if display_curvature_bool == True:
        color_r = np.array(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps)
        rgb_color_arr = np.zeros((len(curvature_values_by_numpy_arr_index_fully_interp_with_end_caps),3))
        rgb_color_arr[:,0] = color_r
        point_cloud = point_containment_tools.create_point_cloud_with_colors_array(threeDdata_array_fully_interpolated_with_end_caps, rgb_color_arr)
        plotting_funcs.plot_geometries(point_cloud)
    
    return curvature_dict




def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi





def compute_structure_triangle_mesh(interp_inter_slice_dist, 
                                    interp_intra_slice_dist,
                                    threeDdata_array_fully_interpolated,
                                    radius_for_normals_estimation,
                                    max_nn_for_normals_estimation
                                    ):
    max_interp_dist = max([interp_inter_slice_dist, interp_intra_slice_dist])
    min_interp_dist = min([interp_inter_slice_dist, interp_intra_slice_dist])
    ball_radii = np.linspace(min_interp_dist, max_interp_dist*2, 3)
    structure_trimesh = meshing_tools.trimesh_reconstruction_ball_pivot(threeDdata_array_fully_interpolated, ball_radii, radius_for_normals_estimation,max_nn_for_normals_estimation)
    watertight_bool = structure_trimesh.is_watertight()

    return structure_trimesh, watertight_bool



def compute_surface_area(mesh):
    """
    Computes the surface area of a triangle mesh object.
    
    Parameters:
        mesh: Open3D TriangleMesh object
    
    Returns:
        surface_area: float, the surface area of the mesh
    """
    # Get the vertices and triangles from the mesh
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Calculate the area of each triangle using cross product
    v0 = vertices[triangles[:, 0]]
    v1 = vertices[triangles[:, 1]]
    v2 = vertices[triangles[:, 2]]
    cross_product = np.cross(v1 - v0, v2 - v0)
    triangle_areas = 0.5 * np.linalg.norm(cross_product, axis=1)
    
    # Compute the total surface area
    surface_area = np.sum(triangle_areas)
    
    return surface_area


def compute_end_caps_area(end_cap_pts_arr,area_voxel_size):
    num_points_in_end_cap = end_cap_pts_arr.shape[0]
    end_cap_area = area_voxel_size*num_points_in_end_cap
    return end_cap_area



def pca_lengths(point_cloud):
    # Perform PCA
    pca = PCA(n_components=3)
    pca.fit(point_cloud)

    # Get the principal components
    components = pca.components_

    # Compute the length of each principal component truncated at maximal points
    lengths = []
    for component in components:
        # Find the maximal point along the component direction
        max_point = np.max(np.dot(point_cloud, component))
        min_point = np.min(np.dot(point_cloud, component))
        # Compute length
        length = max_point - min_point
        lengths.append(length)

    lengths_dict = {"Major": lengths[0],
                    "Minor": lengths[1],
                    "Least": lengths[2]}
    return lengths_dict, components




def draw_oriented_ellipse_point_cloud(centroid_points, axis_lengths, orientation_vectors):
    # Compute the centroid of the input points
    centroid = np.mean(centroid_points, axis=0)

    # Use half of the given diameters as semi-axes lengths
    #axis_lengths = [diameter / 2 for diameter in axis_diameters]

    # Generate points on the oriented ellipse
    num_points = 1000
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = centroid[0] + axis_lengths[0] * np.outer(np.cos(u), np.sin(v))
    y = centroid[1] + axis_lengths[1] * np.outer(np.sin(u), np.sin(v))
    z = centroid[2] + axis_lengths[2] * np.outer(np.ones_like(u), np.cos(v))
    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)

    # Rotate the ellipse according to the orientation vectors
    points = np.dot(points - centroid, orientation_vectors) + centroid

    # Create a point cloud from the generated points
    ellipse_point_cloud = o3d.geometry.PointCloud()
    ellipse_point_cloud.points = o3d.utility.Vector3dVector(points)
    ellipse_point_cloud.paint_uniform_color([1, 0, 0])

    # Create a point cloud from the given centroid points
    centroid_point_cloud = o3d.geometry.PointCloud()
    centroid_point_cloud.points = o3d.utility.Vector3dVector(centroid_points)
    centroid_point_cloud.paint_uniform_color([0, 0, 1]) 

    # Visualize the point clouds
    o3d.visualization.draw_geometries([ellipse_point_cloud, centroid_point_cloud])





def calculate_sphericity(v,a):
    """ 
    v = volume,
    a = surface area
    """
    return ((36*np.pi*(v**2))**(1/3))/a



def calculate_compactness_1(v,a):
    """ 
    v = volume,
    a = surface area
    """
    return v/math.sqrt(np.pi*(a)**3)


def calculate_compactness_2(v,a):
    """ 
    v = volume,
    a = surface area
    """
    return 36*np.pi*((v**2)/(a**3))


def spherical_disproportion(v,a):
    """ 
    v = volume,
    a = surface area
    """
    return a/((36*np.pi*(v**2))**(1/3))











# This function is to reference an array of vectors and add columns to an existing dataframe based on the values of the array and the row is determined by the "Original pt index" column value
# This can be used to reference the biopsy frame points or the test points, etc.
def include_vector_columns_in_dataframe(df, 
                                        vectors, 
                                        reference_column_name, 
                                        new_column_name_x, 
                                        new_column_name_y, 
                                        new_column_name_z):
    """
    Adds 3 new columns to a DataFrame based on an Nx3 array of vectors.
    The mapping is based on the 'Original pt index' column in the DataFrame.
    
    Parameters:
    - df: pandas.DataFrame with a column 'Original pt index'.
    - vectors: Nx3 numpy.array where N is the number of vectors and each vector has 3 elements.
    
    Returns:
    - Modified DataFrame with 3 new columns ('Vector_X', 'Vector_Y', 'Vector_Z') added.
    """
    
    # Check if 'Original pt index' is in the DataFrame
    if 'Original pt index' not in df.columns:
        raise ValueError("DataFrame must contain a column named "+reference_column_name)
    
    # Check if vectors is an Nx3 array
    if vectors.shape[1] != 3:
        raise ValueError("vectors must be an Nx3 array")
    
    # Mapping vectors to new columns based on reference_column_name
    df[new_column_name_x] = vectors[df[reference_column_name].values, 0]
    df[new_column_name_y] = vectors[df[reference_column_name].values, 1]
    df[new_column_name_z] = vectors[df[reference_column_name].values, 2]


    return df