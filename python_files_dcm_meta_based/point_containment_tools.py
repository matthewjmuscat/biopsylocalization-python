import scipy
import numpy as np
import open3d as o3d
from shapely.geometry import Point, Polygon
from bisect import bisect_left
import time
import cuspatial
import cudf
import cupy as cp
import point_containment_tools
import plotting_funcs
import math
import pandas as pd 

#import multiprocess
#import dill
#import pathos, multiprocess
#from pathos.multiprocessing import ProcessingPool
#import dill

def create_point_cloud(data_arr, color = np.array([0,0,0]), random_color = False):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    if random_color == True:
        pcd_color = np.random.uniform(0, 0.9, size=3)
    else:
        pcd_color = color
    point_cloud.paint_uniform_color(pcd_color)
    return point_cloud


def create_point_cloud_with_colors_array(data_arr, rgb_color_arr):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(data_arr)
    point_cloud.colors = o3d.utility.Vector3dVector(rgb_color_arr)
    return point_cloud


def adjacent_slice_delaunay_parallel(parallel_pool, threeD_data_zlsice_list):
    """
    Feed raw data, not interpolated. Data should be a list whose entries are zslices (constant z)
    and each entry (zslice) in the list should be a numpy array of points
    """
    pool = parallel_pool
    adjacent_zslice_threeD_data_list = [(threeD_data_zlsice_list[j],threeD_data_zlsice_list[j+1]) for j in range(0, len(threeD_data_zlsice_list)-1)]
    delaunay_triangulation_obj_zslicewise_list = pool.map(adjacent_zslice_delaunay_triangulation, adjacent_zslice_threeD_data_list)
    #for delaunay_obj in delaunay_triangulation_obj_zslicewise_list:
    #    delaunay_obj.generate_lineset()
    return delaunay_triangulation_obj_zslicewise_list

def adjacent_zslice_delaunay_triangulation(adjacent_zslice_threeD_data_list):
    pcd_color = np.random.uniform(0, 0.7, size=3)
    zslice1 = adjacent_zslice_threeD_data_list[0][0,2]
    zslice2 = adjacent_zslice_threeD_data_list[1][0,2]
    threeDdata_array_adjacent_slices_arr = np.vstack((adjacent_zslice_threeD_data_list[0],adjacent_zslice_threeD_data_list[1]))
    delaunay_triangulation_obj_temp = delaunay_obj(threeDdata_array_adjacent_slices_arr, pcd_color, zslice1, zslice2)
    return delaunay_triangulation_obj_temp



## preliminary test by adjacent zslice delaunay objects



def test_zslice_wise_containment_delaunay_parallel(parallel_pool, delaunay_obj_list, test_points_list):
    pool = parallel_pool
    delaunay_triangle_obj_and_zslice_list = [(x.delaunay_triangulation,x.zslice1,x.zslice2) for x in delaunay_obj_list]
    test_points_arguments_list = [(delaunay_triangle_obj_and_zslice_list,test_points_list[j]) for j in range(len(test_points_list))]
    
    #st = time.time()
    
    test_points_result = pool.starmap(zslice_wise_test_point_containment, test_points_arguments_list)
    
    
    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (delaunay starmap):', elapsed_time, 'seconds')
    #print('___')
    
    return test_points_result
    
    
def zslice_wise_test_point_containment(delauney_tri_object_and_zslice_list,test_point):
    pt_contained = False 
    
    #st = time.time()
    
    for delaunay_obj_index, delaunay_info in enumerate(delauney_tri_object_and_zslice_list):
        #tri.find_simplex(pts) >= 0  is True if point lies within poly)
        delaunay_tri = delaunay_info[0]
        if delaunay_tri.find_simplex(test_point) >= 0:
            zslice1 = delaunay_info[1]
            zslice2 = delaunay_info[2]
            test_pt_color = np.array([0,1,0]) # paint green
            pt_contained = True
            delaunay_obj_contained_in_index = delaunay_obj_index
            break
        else:
            pass            
    if pt_contained == False:
        test_pt_color = np.array([1,0,0]) # paint red
        zslice1 = None
        zslice2 = None
        delaunay_obj_contained_in_index = None
   
    #et = time.time()
    #elapsed_time = et - st
    #print('___')
    #print('\n Execution time (delaunay zslice wise single point):', elapsed_time, 'seconds')
    #print('___')

    return [None,(pt_contained, zslice1, zslice2, delaunay_obj_contained_in_index), test_pt_color, test_point]



## preliminary test by global delaunay object



def test_global_convex_structure_containment_delaunay_parallel(parallel_pool, delaunay_obj, test_points_list):
    pool = parallel_pool
    delaunay_triangle_obj_and_zslice_list = [(delaunay_obj.delaunay_triangulation,delaunay_obj.zslice1,delaunay_obj.zslice2)]
    test_points_arguments_list = [(delaunay_triangle_obj_and_zslice_list,test_points_list[j]) for j in range(len(test_points_list))]
    test_points_result = pool.starmap(convex_structure_global_test_point_containment, test_points_arguments_list)
    return test_points_result


def convex_structure_global_test_point_containment(delauney_tri_object_and_zslice_list,test_point):
    pt_contained = False 
    for delaunay_obj_index, delaunay_info in enumerate(delauney_tri_object_and_zslice_list):
        #tri.find_simplex(pts) >= 0  is True if point lies within poly)
        delaunay_tri = delaunay_info[0]
        if delaunay_tri.find_simplex(test_point) >= 0:
            zslice1 = delaunay_info[1]
            zslice2 = delaunay_info[2]
            test_pt_color = np.array([0,1,0]) # paint green
            pt_contained = True
            delaunay_obj_contained_in_index = delaunay_obj_index
            break
        else:
            pass            
    if pt_contained == False:
        test_pt_color = np.array([1,0,0]) # paint red
        zslice1 = None
        zslice2 = None
        delaunay_obj_contained_in_index = None
    return [None,(pt_contained, zslice1, zslice2, delaunay_obj_contained_in_index), test_pt_color, test_point]



## preliminary test by axis aligned bounding box

def test_global_axis_aligned_box_containment_parallel(parallel_pool, containment_structure_pts_arr, test_points_list):
    pool = parallel_pool
    test_points_arguments_list = [(containment_structure_pts_arr,test_points_list[j]) for j in range(len(test_points_list))]
    test_points_result = pool.starmap(axis_aligned_bounding_box_test_point_containment, test_points_arguments_list)
    return test_points_result


def axis_aligned_bounding_box_test_point_containment(containment_structure_pts_arr,test_point):
    containment_structure_pcd = create_point_cloud(containment_structure_pts_arr)
    axis_aligned_bounding_box = containment_structure_pcd.get_axis_aligned_bounding_box()

    pt_contained = None 
    max_XYZ_bound = axis_aligned_bounding_box.get_max_bound()
    min_XYZ_bound = axis_aligned_bounding_box.get_min_bound()

    max_X = max_XYZ_bound[0]
    max_Y = max_XYZ_bound[1]
    max_Z = max_XYZ_bound[2]

    min_X = min_XYZ_bound[0]
    min_Y = min_XYZ_bound[1]
    min_Z = min_XYZ_bound[2]

    tp_X = test_point[0]
    tp_Y = test_point[1]
    tp_Z = test_point[2]
    
    # test containment
    if (min_X <= tp_X and tp_X <= max_X) and (min_Y <= tp_Y and tp_Y <= max_Y) and (min_Z <= tp_Z and tp_Z <= max_Z):
        test_pt_color = np.array([0,1,0]) # paint green
        pt_contained = True
    else:
        test_pt_color = np.array([1,0,0]) # paint red 
        pt_contained = False         
       
    return [None,(pt_contained,), test_pt_color, test_point]






def plane_point_in_polygon_concave(test_points_results,interslice_interpolation_information, test_pts_point_cloud):
    test_points_contained_in_delaunay = [(test_result_org_index,test_result) for test_result_org_index,test_result in enumerate(test_points_results) if test_result[1][0]==True]
    for test_point_info in test_points_contained_in_delaunay:
        test_point_original_index = test_point_info[0]
        test_point_data = test_point_info[1]
        test_point = test_point_data[3]
        test_point_2d = test_point[0:2]
        test_point_zval = test_point[2]
        shapely_test_point = Point(test_point_2d)
        interpolated_zvlas_list = interslice_interpolation_information.zslice_vals_after_interpolation_list    
        nearest_interpolated_zslice_index, nearest_interpolated_zslice_val = take_closest(interpolated_zvlas_list, test_point_zval)
        nearest_zslice = interslice_interpolation_information.interpolated_pts_list[nearest_interpolated_zslice_index]
        nearest_zslice_2d = nearest_zslice[:,0:2]
        nearest_zslice_shapely_polygon = Polygon(nearest_zslice_2d)
        point_contained_in_nearest_zslice_bool = shapely_test_point.within(nearest_zslice_shapely_polygon)
        test_point_concave_zslice_data = (point_contained_in_nearest_zslice_bool,
            nearest_interpolated_zslice_val, nearest_interpolated_zslice_index)
        if point_contained_in_nearest_zslice_bool == True:
            pass
        else:
            test_points_results[test_point_original_index][2] = np.array([1,0,0],dtype=float)
        test_points_results[test_point_original_index][0] = test_point_concave_zslice_data
    
    test_pts_point_cloud_concave_zlsice_updated = o3d.geometry.PointCloud()
    test_pts_point_cloud_concave_zlsice_updated.points = test_pts_point_cloud.points 
    num_points = len(test_pts_point_cloud_concave_zlsice_updated.points)
    test_pts_colors_zslice_concave_updated = np.empty([num_points,3], dtype=float)
    
    for index,result in enumerate(test_points_results):
        test_pts_colors_zslice_concave_updated[index] = result[2]
    test_pts_point_cloud_concave_zlsice_updated.colors = o3d.utility.Vector3dVector(test_pts_colors_zslice_concave_updated)
    return test_points_results, test_pts_point_cloud_concave_zlsice_updated



def cuspatial_points_contained(polygons_geoseries,
                               test_points_geoseries, 
                               test_points_array, 
                               nearest_zslices_indices_arr,
                               nearest_zslices_vals_arr,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval,  
                               num_sample_pts_in_bx,
                               num_MC_containment_simulations,
                               structure_info
                               ):
    
    num_zslices = len(polygons_geoseries)
    total_num_points = len(test_points_geoseries) # note that this is the total number of points num_MC_containment_sims*num_sampled_pts_in_bx+1, the +1 is because nominal position is included
    current_index = 0
    next_index = 0
    results_dataframes_list = []
    while next_index < num_zslices:
        if num_zslices - current_index > 30:
            next_index = current_index + 30
        else:
            next_index = num_zslices
        containment_results_grand_dataframe_polygon_subset = cuspatial.point_in_polygon(test_points_geoseries,
                                        polygons_geoseries[current_index:next_index]                   
                                        )
        results_dataframes_list.append(containment_results_grand_dataframe_polygon_subset)
        current_index = next_index
        
    containment_results_grand_dataframe = cudf.concat(results_dataframes_list,axis=1)

    #contain_bool_by_point_list = [containment_results_grand_dataframe.values[list_ind % num_sample_pts_in_bx, polygon_ind] for list_ind,polygon_ind in enumerate(nearest_zslices_indices_arr)]
    #contain_bool_arr = cp.array(contain_bool_by_point_list)
    points_arr_org_index = np.array([list_ind % num_sample_pts_in_bx for list_ind in range(total_num_points)])
    points_arr_index = np.arange(0,total_num_points)
    contain_bool_arr_step_1 = containment_results_grand_dataframe.to_cupy()[points_arr_index,nearest_zslices_indices_arr]
    # further set points that are outside z-range to False
    pts_contained_below_max_zval = cp.array((test_points_array[:,2] < non_bx_struct_max_zval))
    pts_contained_above_min_zval = cp.array((test_points_array[:,2] > non_bx_struct_min_zval))
    pts_contained_between_zvals = cp.logical_and(pts_contained_below_max_zval,pts_contained_above_min_zval)
    contain_bool_arr = cp.logical_and(contain_bool_arr_step_1,pts_contained_between_zvals)
    contain_color_arr = color_by_bool_by_arr(cp.asnumpy(contain_bool_arr))
    contain_color_arr[0:num_sample_pts_in_bx,2] = 1 # set the nominal points to turn on blue for contained color, therefore False = pink and True = Cyan
    trial_number_list = [int(i) for i in range(num_MC_containment_simulations+1) for j in range(num_sample_pts_in_bx)] # Note that the nominal position is indicated by Trial num = 0
    results_dictionary = {"Relative structure ROI": structure_info[0],
                          "Relative structure type": structure_info[1],
                          "Relative structure index": structure_info[3],
                          "Original pt index": points_arr_org_index,
                          "Pt contained bool": contain_bool_arr,
                          "Nearest zslice zval": nearest_zslices_vals_arr,
                          "Nearest zslice index": nearest_zslices_indices_arr,
                          "Pt clr R": contain_color_arr[:,0],
                          "Pt clr G": contain_color_arr[:,1],
                          "Pt clr B": contain_color_arr[:,2],
                          "Test pt X": test_points_array[:,0],
                          "Test pt Y": test_points_array[:,1],
                          "Test pt Z": test_points_array[:,2],
                          "Trial num": trial_number_list
                          }

    
    grand_cudf_dataframe = cudf.DataFrame.from_dict(results_dictionary)

    return grand_cudf_dataframe


def cuspatial_points_contained_mc_sim_cupy_pandas(polygons_geoseries,
                               test_points_geoseries, 
                               test_points_array, 
                               nearest_zslices_indices_arr,
                               nearest_zslices_vals_arr,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval,  
                               num_sample_pts_in_bx,
                               num_MC_containment_simulations,
                               patientUID,
                               biopsy_structure_info,
                               structure_info,
                               layout_groups,
                               live_display,
                               progress_bar_level_task_obj,
                               indeterminate_bar_level_task_obj,
                               upper_limit_size_input = 1e9,
                               matrix_key = None
                               ):
    
    num_zslices = len(polygons_geoseries)
    total_num_points = len(test_points_geoseries) # note that this is the total number of points num_MC_containment_sims*num_sampled_pts_in_bx+1, the +1 is because nominal position is included
    upper_limit_size = upper_limit_size_input
    if num_zslices*total_num_points > upper_limit_size:
        ###
        indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~Very large array, building prelims (1)", total = None)
        ###

        contain_bool_arr_step_1 = cp.empty(total_num_points, dtype = cp.bool_)
        chunk_row_size = math.floor(upper_limit_size/num_zslices)
        num_point_chunks = math.ceil(total_num_points/chunk_row_size)
        next_pt_index = 0
        current_pt_index = 0

        ###
        indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
        ###


        task_description = "[cyan]Processing point chunks"
        chunk_task = progress_bar_level_task_obj.add_task(task_description, total=num_point_chunks)
        while next_pt_index < total_num_points:
            if total_num_points - current_pt_index > chunk_row_size:
                next_pt_index = current_pt_index + chunk_row_size
            else:
                next_pt_index = total_num_points

            current_index = 0
            next_index = 0

            ###
            indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~Building pointschunk cupy array (2)", total = None)
            ###

            containment_results_grand_cupy_array_pointschunk = cp.empty((next_pt_index-current_pt_index,num_zslices), dtype = cp.bool_)

            ###
            indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
            ###

            num_slice_chunks = math.ceil(num_zslices/30)
            slice_task_description = "[cyan]Processing slicechunk"
            slice_task = progress_bar_level_task_obj.add_task(slice_task_description, total=num_slice_chunks)
            while next_index < num_zslices:
                if num_zslices - current_index > 30:
                    next_index = current_index + 30
                else:
                    next_index = num_zslices

                ###
                indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~cuspatial (3)", total = None)
                ###

                containment_results_grand_cupy_array_pointschunk_polygonchunk = cuspatial.point_in_polygon(test_points_geoseries[current_pt_index:next_pt_index],
                                                polygons_geoseries[current_index:next_index]                   
                                                ).to_cupy()
                

                ###
                indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
                ###
                
                ###
                indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~Assigning (4)", total = None)
                ###

                containment_results_grand_cupy_array_pointschunk[:,current_index:next_index] = containment_results_grand_cupy_array_pointschunk_polygonchunk
                
                ###
                indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
                ###

                del containment_results_grand_cupy_array_pointschunk_polygonchunk
                current_index = next_index

                progress_bar_level_task_obj.update(slice_task, advance=1)
            progress_bar_level_task_obj.update(slice_task, visible=False)

            ###
            indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~Creating and assigning (5)", total = None)
            ###

            points_arr_index = cp.arange(0,next_pt_index-current_pt_index, dtype = cp.int64)
            contain_bool_arr_step_1_points_chunk = containment_results_grand_cupy_array_pointschunk[points_arr_index,nearest_zslices_indices_arr[current_pt_index:next_pt_index]]
            del containment_results_grand_cupy_array_pointschunk
            del points_arr_index
            contain_bool_arr_step_1[current_pt_index:next_pt_index] = contain_bool_arr_step_1_points_chunk
            del contain_bool_arr_step_1_points_chunk

            ###
            indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
            ###
            
            current_pt_index = next_pt_index

            progress_bar_level_task_obj.update(chunk_task, advance=1)
        progress_bar_level_task_obj.update(chunk_task, visible=False)


    else: 
        current_index = 0
        next_index = 0
        ###
        indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~Small array, creating cupy arr (1)", total = None)
        ###
        
        containment_results_grand_cupy_array = cp.empty((total_num_points,num_zslices), dtype = cp.bool_)

        ###
        indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
        ###

        num_slice_chunks = math.ceil(num_zslices/30)
        slice_task_description = "[cyan]Processing slicechunk"
        slice_task = progress_bar_level_task_obj.add_task(slice_task_description, total=num_slice_chunks)
        while next_index < num_zslices:
            if num_zslices - current_index > 30:
                next_index = current_index + 30
            else:
                next_index = num_zslices

            ###
            indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~cuspatial (2)", total = None)
            ###

            containment_results_grand_cupy_array_polygon_subset = cuspatial.point_in_polygon(test_points_geoseries,
                                            polygons_geoseries[current_index:next_index]                   
                                            ).to_cupy()
            
            ###
            indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
            ###

            ###
            indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~assigning (3)", total = None)
            ###
            
            containment_results_grand_cupy_array[:,current_index:next_index] = containment_results_grand_cupy_array_polygon_subset

            ###
            indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
            ###

            current_index = next_index

            progress_bar_level_task_obj.update(slice_task, advance=1)
        progress_bar_level_task_obj.update(slice_task, visible=False)

        ###
        indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~ Creating and assigning (4)", total = None)
        ###

        points_arr_index = cp.arange(0,total_num_points, dtype=cp.int64)
        contain_bool_arr_step_1 = containment_results_grand_cupy_array[points_arr_index,nearest_zslices_indices_arr]
        del containment_results_grand_cupy_array
        del points_arr_index

        ###
        indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
        ###

    points_arr_org_index_template = cp.array(range(num_sample_pts_in_bx))
    points_arr_org_index = cp.tile(points_arr_org_index_template,num_MC_containment_simulations+1) # the +1 is because of the nominal position!
    del points_arr_org_index_template

    ###
    indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~ logic (5)", total = None)
    ###

    # further set points that are outside z-range to False
    pts_contained_below_max_zval = cp.array((test_points_array[:,2] < non_bx_struct_max_zval))
    pts_contained_above_min_zval = cp.array((test_points_array[:,2] > non_bx_struct_min_zval))
    pts_contained_between_zvals = cp.logical_and(pts_contained_below_max_zval,pts_contained_above_min_zval)
    del pts_contained_below_max_zval
    del pts_contained_above_min_zval
    contain_bool_arr = cp.logical_and(contain_bool_arr_step_1,pts_contained_between_zvals)
    del contain_bool_arr_step_1
    del pts_contained_between_zvals

    ###
    indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
    ###

    ###
    indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~coloring (6)", total = None)
    ###

    contain_color_arr = color_by_bool_array_cupy_fast(contain_bool_arr)
    contain_color_arr[0:num_sample_pts_in_bx,2] = 1 # set the nominal points to turn on blue for contained color, therefore False = pink and True = Cyan
    
    ###
    indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
    ###
    

    ###
    indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~creating (7)", total = None)
    ###

    #trial_number_list = [int(i) for i in range(num_MC_containment_simulations+1) for j in range(num_sample_pts_in_bx)] # Note that the nominal position is indicated by Trial num = 0
    trial_number_template_arr = cp.array(range(num_MC_containment_simulations+1))
    trial_number_arr = cp.repeat(trial_number_template_arr, num_sample_pts_in_bx)

    ###
    indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
    ###

    ###
    indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~creating dictionary (8)", total = None)
    ###

    if matrix_key == None:
        results_dictionary = {"Patient ID": patientUID,
                            "Bx ID": biopsy_structure_info["Structure ID"],
                            "Biopsy refnum": biopsy_structure_info["Dicom ref num"],
                            "Bx index": biopsy_structure_info["Index number"],
                            "Relative structure ROI": structure_info[0],
                            "Relative structure type": structure_info[1],
                            "Relative structure index": structure_info[3],
                            "Original pt index": points_arr_org_index.get(),
                            "Pt contained bool": contain_bool_arr.get(),
                            "Nearest zslice zval": nearest_zslices_vals_arr,
                            "Nearest zslice index": nearest_zslices_indices_arr,
                            "Pt clr R": contain_color_arr[:,0].get(),
                            "Pt clr G": contain_color_arr[:,1].get(),
                            "Pt clr B": contain_color_arr[:,2].get(),
                            "Test pt X": test_points_array[:,0],
                            "Test pt Y": test_points_array[:,1],
                            "Test pt Z": test_points_array[:,2],
                            "Trial num": trial_number_arr.get()
                            }
    else:
        results_dictionary = {"Patient ID": patientUID,
                            "Bx ID": biopsy_structure_info["Structure ID"],
                            "Biopsy refnum": biopsy_structure_info["Dicom ref num"],
                            "Bx index": biopsy_structure_info["Index number"],
                            "Relative structure ROI": structure_info[0],
                            "Relative structure type": structure_info[1],
                            "Relative structure index": structure_info[3],
                            "Original pt index": points_arr_org_index.get(),
                            "Matrix key": matrix_key,
                            "Pt contained bool": contain_bool_arr.get(),
                            "Nearest zslice zval": nearest_zslices_vals_arr,
                            "Nearest zslice index": nearest_zslices_indices_arr,
                            "Pt clr R": contain_color_arr[:,0].get(),
                            "Pt clr G": contain_color_arr[:,1].get(),
                            "Pt clr B": contain_color_arr[:,2].get(),
                            "Test pt X": test_points_array[:,0],
                            "Test pt Y": test_points_array[:,1],
                            "Test pt Z": test_points_array[:,2],
                            "Trial num": trial_number_arr.get()
                            }
        
    ###
    indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
    ###

    ###
    indeterminate_task = indeterminate_bar_level_task_obj.add_task("[cyan]~~creating dataframe (9)", total = None)
    ###

    #grand_pandas_dataframe = cudf.DataFrame.from_dict(results_dictionary).to_pandas()
    grand_pandas_dataframe = pd.DataFrame.from_dict(results_dictionary)

    ###
    indeterminate_bar_level_task_obj.update(indeterminate_task, visible = False)
    ###

    return grand_pandas_dataframe, live_display


def cuspatial_points_contained_FANOVA(polygons_geoseries,
                               test_points_geoseries, 
                               test_points_array, 
                               nearest_zslices_indices_arr,
                               nearest_zslices_vals_arr,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval,  
                               num_sample_pts_in_bx,
                               num_FANOVA_containment_simulations,
                               structure_info,
                               matrix_key
                               ):
    
    num_zslices = len(polygons_geoseries)
    total_num_points = len(test_points_geoseries) # note that this is the total number of points num_FANOVA_containment_sims*num_sampled_pts_in_bx
    current_index = 0
    next_index = 0
    results_dataframes_list = []
    while next_index < num_zslices:
        if num_zslices - current_index > 30:
            next_index = current_index + 30
        else:
            next_index = num_zslices
        containment_results_grand_dataframe_polygon_subset = cuspatial.point_in_polygon(test_points_geoseries,
                                        polygons_geoseries[current_index:next_index]                   
                                        )
        results_dataframes_list.append(containment_results_grand_dataframe_polygon_subset)
        current_index = next_index
        
    containment_results_grand_dataframe = cudf.concat(results_dataframes_list,axis=1)

    #contain_bool_by_point_list = [containment_results_grand_dataframe.values[list_ind % num_sample_pts_in_bx, polygon_ind] for list_ind,polygon_ind in enumerate(nearest_zslices_indices_arr)]
    #contain_bool_arr = cp.array(contain_bool_by_point_list)
    points_arr_org_index = np.array([list_ind % num_sample_pts_in_bx for list_ind in range(total_num_points)])
    points_arr_index = np.arange(0,total_num_points)
    contain_bool_arr_step_1 = containment_results_grand_dataframe.to_cupy()[points_arr_index,nearest_zslices_indices_arr]
    # further set points that are outside z-range to False
    pts_contained_below_max_zval = cp.array((test_points_array[:,2] < non_bx_struct_max_zval))
    pts_contained_above_min_zval = cp.array((test_points_array[:,2] > non_bx_struct_min_zval))
    pts_contained_between_zvals = cp.logical_and(pts_contained_below_max_zval,pts_contained_above_min_zval)
    contain_bool_arr = cp.logical_and(contain_bool_arr_step_1,pts_contained_between_zvals)
    contain_color_arr = color_by_bool_by_arr(cp.asnumpy(contain_bool_arr))
    #contain_color_arr[0:num_sample_pts_in_bx,2] = 1 # set the nominal points to turn on blue for contained color, therefore False = pink and True = Cyan
    trial_number_list = [int(i+1) for i in range(num_FANOVA_containment_simulations) for j in range(num_sample_pts_in_bx)]
    results_dictionary = {"Relative structure ROI": structure_info[0],
                          "Relative structure type": structure_info[1],
                          "Relative structure index": structure_info[3],
                          "Matrix key": matrix_key,
                          "Original pt index": points_arr_org_index,
                          "Pt contained bool": contain_bool_arr,
                          "Nearest zslice zval": nearest_zslices_vals_arr,
                          "Nearest zslice index": nearest_zslices_indices_arr,
                          "Pt clr R": contain_color_arr[:,0],
                          "Pt clr G": contain_color_arr[:,1],
                          "Pt clr B": contain_color_arr[:,2],
                          "Test pt X": test_points_array[:,0],
                          "Test pt Y": test_points_array[:,1],
                          "Test pt Z": test_points_array[:,2],
                          "Trial num": trial_number_list
                          }

    
    grand_cudf_dataframe = cudf.DataFrame.from_dict(results_dictionary)

    return grand_cudf_dataframe



def cuspatial_points_contained_generic_numpy_pandas(polygons_geoseries,
                               test_points_geoseries, 
                               test_points_array, 
                               nearest_zslices_indices_arr,
                               nearest_zslices_vals_arr,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval,
                               structure_info,
                               layout_groups,
                               live_display,
                               progress_bar_level_task_obj,
                               upper_limit_size_input = 1e9
                               ):
    
    num_zslices = len(polygons_geoseries)
    total_num_points = len(test_points_geoseries)
    upper_limit_size = upper_limit_size_input
    if num_zslices*total_num_points > upper_limit_size:
        
        contain_bool_arr_step_1 = np.empty(total_num_points)
        chunk_row_size = math.floor(upper_limit_size/num_zslices)
        num_point_chunks = math.ceil(total_num_points/chunk_row_size)
        next_pt_index = 0
        current_pt_index = 0

        task_description = "[cyan]Processing point chunks"
        chunk_task = progress_bar_level_task_obj.add_task(task_description, total=num_point_chunks)
        while next_pt_index < total_num_points:
            if total_num_points - current_pt_index > chunk_row_size:
                next_pt_index = current_pt_index + chunk_row_size
            else:
                next_pt_index = total_num_points

            current_index = 0
            next_index = 0

            containment_results_grand_numpy_array_pointschunk = np.empty((next_pt_index-current_pt_index,num_zslices))
            num_slice_chunks = math.ceil(num_zslices/30)
            slice_task_description = "[cyan]Processing slicechunk"
            slice_task = progress_bar_level_task_obj.add_task(slice_task_description, total=num_slice_chunks)
            while next_index < num_zslices:
                if num_zslices - current_index > 30:
                    next_index = current_index + 30
                else:
                    next_index = num_zslices

                containment_results_grand_numpy_array_pointschunk_polygonchunk = cuspatial.point_in_polygon(test_points_geoseries[current_pt_index:next_pt_index],
                                                polygons_geoseries[current_index:next_index]                   
                                                ).to_numpy()
                
                containment_results_grand_numpy_array_pointschunk[:,current_index:next_index] = containment_results_grand_numpy_array_pointschunk_polygonchunk

                current_index = next_index

                progress_bar_level_task_obj.update(slice_task, advance=1)
            progress_bar_level_task_obj.update(slice_task, visible=False)

            points_arr_index = np.arange(0,next_pt_index-current_pt_index, dtype=np.int64)
            contain_bool_arr_step_1_points_chunk = containment_results_grand_numpy_array_pointschunk[points_arr_index,nearest_zslices_indices_arr[current_pt_index:next_pt_index]]
            contain_bool_arr_step_1[current_pt_index:next_pt_index] = np.array(contain_bool_arr_step_1_points_chunk)
            
            current_pt_index = next_pt_index

            progress_bar_level_task_obj.update(chunk_task, advance=1)
        progress_bar_level_task_obj.update(chunk_task, visible=False)


    else: 
        current_index = 0
        next_index = 0
        #results_dataframes_list = []
        containment_results_grand_numpy_array = np.empty((total_num_points,num_zslices))

        num_slice_chunks = math.ceil(num_zslices/30)
        slice_task_description = "[cyan]Processing slicechunk"
        slice_task = progress_bar_level_task_obj.add_task(slice_task_description, total=num_slice_chunks)
        while next_index < num_zslices:
            if num_zslices - current_index > 30:
                next_index = current_index + 30
            else:
                next_index = num_zslices

            containment_results_grand_numpy_array_polygon_subset = cuspatial.point_in_polygon(test_points_geoseries,
                                            polygons_geoseries[current_index:next_index]                   
                                            ).to_numpy()

            containment_results_grand_numpy_array[:,current_index:next_index] = containment_results_grand_numpy_array_polygon_subset

            current_index = next_index

            progress_bar_level_task_obj.update(slice_task, advance=1)
        progress_bar_level_task_obj.update(slice_task, visible=False)

        points_arr_index = np.arange(0,total_num_points, dtype=np.int64)

        contain_bool_arr_step_1 = containment_results_grand_numpy_array[points_arr_index,nearest_zslices_indices_arr]


    # further set points that are outside z-range to False
    pts_contained_below_max_zval = np.array((test_points_array[:,2] < non_bx_struct_max_zval))
    pts_contained_above_min_zval = np.array((test_points_array[:,2] > non_bx_struct_min_zval))
    pts_contained_between_zvals = np.logical_and(pts_contained_below_max_zval,pts_contained_above_min_zval)
    contain_bool_arr = np.logical_and(contain_bool_arr_step_1,pts_contained_between_zvals)
    contain_color_arr = color_by_bool_array_numpy_fast(contain_bool_arr)
    
    results_dictionary = {"Relative structure ROI": structure_info["Structure ID"],
                        "Relative structure type": structure_info["Struct ref type"],
                        "Relative structure index": structure_info["Index number"],
                        "Pt contained bool": contain_bool_arr,
                        "Nearest zslice zval": nearest_zslices_vals_arr,
                        "Nearest zslice index": nearest_zslices_indices_arr,
                        "Pt clr R": contain_color_arr[:,0],
                        "Pt clr G": contain_color_arr[:,1],
                        "Pt clr B": contain_color_arr[:,2],
                        "Test pt X": test_points_array[:,0],
                        "Test pt Y": test_points_array[:,1],
                        "Test pt Z": test_points_array[:,2]
                        }


    grand_pandas_dataframe = pd.DataFrame.from_dict(results_dictionary)

    return grand_pandas_dataframe, live_display




def cuspatial_points_contained_generic_cupy_pandas(polygons_geoseries,
                               test_points_geoseries, 
                               test_points_array, 
                               nearest_zslices_indices_arr,
                               nearest_zslices_vals_arr,
                               non_bx_struct_max_zval,
                               non_bx_struct_min_zval,
                               structure_info,
                               layout_groups,
                               live_display,
                               progress_bar_level_task_obj,
                               upper_limit_size_input = 1e9
                               ):
    
    num_zslices = len(polygons_geoseries)
    total_num_points = len(test_points_geoseries)
    upper_limit_size = upper_limit_size_input
    if num_zslices*total_num_points > upper_limit_size:
        
        contain_bool_arr_step_1 = cp.empty(total_num_points, dtype = cp.bool_)
        chunk_row_size = math.floor(upper_limit_size/num_zslices)
        num_point_chunks = math.ceil(total_num_points/chunk_row_size)
        next_pt_index = 0
        current_pt_index = 0

        task_description = "[cyan]Processing point chunks"
        chunk_task = progress_bar_level_task_obj.add_task(task_description, total=num_point_chunks)
        while next_pt_index < total_num_points:
            if total_num_points - current_pt_index > chunk_row_size:
                next_pt_index = current_pt_index + chunk_row_size
            else:
                next_pt_index = total_num_points

            current_index = 0
            next_index = 0

            containment_results_grand_numpy_array_pointschunk = cp.empty((next_pt_index-current_pt_index,num_zslices), dtype = cp.bool_)
            num_slice_chunks = math.ceil(num_zslices/30)
            slice_task_description = "[cyan]Processing slicechunk"
            slice_task = progress_bar_level_task_obj.add_task(slice_task_description, total=num_slice_chunks)
            while next_index < num_zslices:
                if num_zslices - current_index > 30:
                    next_index = current_index + 30
                else:
                    next_index = num_zslices

                containment_results_grand_numpy_array_pointschunk_polygonchunk = cuspatial.point_in_polygon(test_points_geoseries[current_pt_index:next_pt_index],
                                                polygons_geoseries[current_index:next_index]                   
                                                ).to_cupy()
                
                containment_results_grand_numpy_array_pointschunk[:,current_index:next_index] = containment_results_grand_numpy_array_pointschunk_polygonchunk
                del containment_results_grand_numpy_array_pointschunk_polygonchunk
                current_index = next_index

                progress_bar_level_task_obj.update(slice_task, advance=1)
            progress_bar_level_task_obj.update(slice_task, visible=False)

            points_arr_index = cp.arange(0,next_pt_index-current_pt_index, dtype=cp.int64)
            contain_bool_arr_step_1_points_chunk = containment_results_grand_numpy_array_pointschunk[points_arr_index,nearest_zslices_indices_arr[current_pt_index:next_pt_index]]
            del containment_results_grand_numpy_array_pointschunk
            contain_bool_arr_step_1[current_pt_index:next_pt_index] = contain_bool_arr_step_1_points_chunk
            del contain_bool_arr_step_1_points_chunk
            
            current_pt_index = next_pt_index

            progress_bar_level_task_obj.update(chunk_task, advance=1)
        progress_bar_level_task_obj.update(chunk_task, visible=False)


    else: 
        current_index = 0
        next_index = 0
        #results_dataframes_list = []
        containment_results_grand_numpy_array = cp.empty((total_num_points,num_zslices), dtype = cp.bool_)

        num_slice_chunks = math.ceil(num_zslices/30)
        slice_task_description = "[cyan]Processing slicechunk"
        slice_task = progress_bar_level_task_obj.add_task(slice_task_description, total=num_slice_chunks)
        while next_index < num_zslices:
            if num_zslices - current_index > 30:
                next_index = current_index + 30
            else:
                next_index = num_zslices

            containment_results_grand_numpy_array_polygon_subset = cuspatial.point_in_polygon(test_points_geoseries,
                                            polygons_geoseries[current_index:next_index]                   
                                            ).to_cupy()

            containment_results_grand_numpy_array[:,current_index:next_index] = containment_results_grand_numpy_array_polygon_subset
            del containment_results_grand_numpy_array_polygon_subset

            current_index = next_index

            progress_bar_level_task_obj.update(slice_task, advance=1)
        progress_bar_level_task_obj.update(slice_task, visible=False)

        points_arr_index = cp.arange(0,total_num_points, dtype=cp.int64)

        contain_bool_arr_step_1 = containment_results_grand_numpy_array[points_arr_index,nearest_zslices_indices_arr]
        del containment_results_grand_numpy_array


    # further set points that are outside z-range to False
    pts_contained_below_max_zval = cp.array((test_points_array[:,2] < non_bx_struct_max_zval))
    pts_contained_above_min_zval = cp.array((test_points_array[:,2] > non_bx_struct_min_zval))
    pts_contained_between_zvals = cp.logical_and(pts_contained_below_max_zval,pts_contained_above_min_zval)
    del pts_contained_below_max_zval
    del pts_contained_above_min_zval
    contain_bool_arr = cp.logical_and(contain_bool_arr_step_1,pts_contained_between_zvals)
    del contain_bool_arr_step_1
    del pts_contained_between_zvals
    contain_color_arr = color_by_bool_array_cupy_fast(contain_bool_arr)
    
    results_dictionary = {"Relative structure ROI": structure_info["Structure ID"],
                        "Relative structure type": structure_info["Struct ref type"],
                        "Relative structure index": structure_info["Index number"],
                        "Pt contained bool": contain_bool_arr.get(),
                        "Nearest zslice zval": nearest_zslices_vals_arr,
                        "Nearest zslice index": nearest_zslices_indices_arr,
                        "Pt clr R": contain_color_arr[:,0].get(),
                        "Pt clr G": contain_color_arr[:,1].get(),
                        "Pt clr B": contain_color_arr[:,2].get(),
                        "Test pt X": test_points_array[:,0],
                        "Test pt Y": test_points_array[:,1],
                        "Test pt Z": test_points_array[:,2]
                        }

    #grand_pandas_dataframe = cudf.DataFrame.from_dict(results_dictionary).to_pandas()
    grand_pandas_dataframe = pd.DataFrame.from_dict(results_dictionary)

    return grand_pandas_dataframe, live_display




def color_by_bool_by_arr(contain_bool_arr):
    color_arr = np.empty([len(contain_bool_arr),3])
    for index,element_bool in enumerate(contain_bool_arr):
        if element_bool == True:
            color = np.array([0,1,0]) # green
        else:
            color = np.array([1,0,0]) # red
        color_arr[index,:] = color
    return color_arr


def color_by_bool(contain_bool):
    if contain_bool == True:
        color_arr = np.array([0,1,0]) # green
    else:
        color_arr = np.array([1,0,0]) # red
    return color_arr

def color_by_bool_array_cupy_fast(cupy_bool_val_arr):
    color_arr = cp.empty((cupy_bool_val_arr.shape[0],3))
    
    green_arr = cupy_bool_val_arr.astype(int)
    red_arr = 1- green_arr
    blue_arr = cp.zeros((cupy_bool_val_arr.shape[0]))
    
    color_arr[:,0] = red_arr
    color_arr[:,1] = green_arr
    color_arr[:,2] = blue_arr

    return color_arr


def color_by_bool_array_numpy_fast(cupy_bool_val_arr):
    color_arr = np.empty((cupy_bool_val_arr.shape[0],3))
    
    green_arr = cupy_bool_val_arr.astype(int)
    red_arr = 1- green_arr
    blue_arr = np.zeros((cupy_bool_val_arr.shape[0]))
    
    color_arr[:,0] = red_arr
    color_arr[:,1] = green_arr
    color_arr[:,2] = blue_arr

    return color_arr

def take_closest(myList_org, myNumber):
    """
    Assumes myList is sorted. Returns index of closest value and the closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    myList = myList_org.copy()
    if myList[0] > myList[1]:
        myList.reverse()
        list_reversed = True
    else:
        list_reversed = False
    if list_reversed == False:
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return 0, myList[0]
        if pos == len(myList):
            return pos-1, myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return pos, after
        else:
            return pos - 1, before
    elif list_reversed == True:
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return len(myList)-1, myList[0]
        if pos == len(myList):
            return 0, myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return len(myList) - 1 - pos, after
        else:
            return len(myList) - pos, before

def take_closest_cp(myList_org, myNumber):
    """
    Assumes myList is sorted. Returns index of closest value and the closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    myList = myList_org.tolist().copy()
    if myList[0] > myList[1]:
        myList.reverse()
        list_reversed = True
    else:
        list_reversed = False
    if list_reversed == False:
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return 0, myList[0]
        if pos == len(myList):
            return pos-1, myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return pos, after
        else:
            return pos - 1, before
    elif list_reversed == True:
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return len(myList)-1, myList[0]
        if pos == len(myList):
            return 0, myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return len(myList) - 1 - pos, after
        else:
            return len(myList) - pos, before
        
def take_closest_index_only(myList_org, myNumber):
    """
    Assumes myList is sorted. Returns index of closest value and the closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    myList = myList_org.copy()
    if myList[0] > myList[1]:
        myList.reverse()
        list_reversed = True
    else:
        list_reversed = False
    if list_reversed == False:
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return 0
        if pos == len(myList):
            return pos-1
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return pos
        else:
            return pos - 1
    elif list_reversed == True:
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return len(myList)-1
        if pos == len(myList):
            return 0
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return len(myList) - 1 - pos
        else:
            return len(myList) - pos
        


def take_closest_array_input(myList, myArray):
    """
    Assumes myList is sorted. Returns index of closest value and the closest value to myNumber.

    If two numbers are equally close, return the smallest number. Takes an array as input and vectorizes the 'take_closest_index_only' function. 

    Returns an array of the indices of the original list that contains the value that is closest to each element of the given array.
    """

    take_closest_index_only_vectorized = np.vectorize(take_closest, excluded=['myList_org'])
    closest_vals_indices_array, closest_vals_array = take_closest_index_only_vectorized(myList_org = myList,myNumber = myArray)
    return closest_vals_indices_array, closest_vals_array


def take_closest_array_input_cp(myList, myArray):
    """
    Assumes myList is sorted. Returns index of closest value and the closest value to myNumber.

    If two numbers are equally close, return the smallest number. Takes an array as input and vectorizes the 'take_closest_index_only' function. 

    Returns an array of the indices of the original list that contains the value that is closest to each element of the given array.
    """
    mylistTiled = np.tile(myList,(myArray.shape[0],1))
    take_closest_index_only_vectorized = cp.vectorize(take_closest_cp)
    closest_vals_indices_array, closest_vals_array = take_closest_index_only_vectorized(mylistTiled,myArray)
    return closest_vals_indices_array, closest_vals_array


def convex_bx_structure_global_test_point_containment(global_delaunay_tri,test_point):
    if global_delaunay_tri.find_simplex(test_point) >= 0:
        pt_contained = True
    else:
        pt_contained = False            
    return pt_contained


def take_closest_cupy(indexed_array, testing_values_array):

    "Note that: In case of multiple occurrences of the minimum values, the indices corresponding to the first occurrence are returned."

    indexed_array_cupy = cp.array(indexed_array)
    testing_values_array_cupy = cp.array(testing_values_array)

    indexed_array_cupy_tiled = cp.tile(indexed_array_cupy, (testing_values_array_cupy.shape[0],1)) 

    dist_array = cp.absolute(indexed_array_cupy_tiled.transpose()-testing_values_array_cupy).transpose()

    del indexed_array_cupy_tiled
    
    # THE RESULTS INDICES ARRAY MUST HAVE AN INTEGER DATA TYPE!
    results_indices = cp.asnumpy(dist_array.argmin(axis=1).astype(cp.int64))

    results_nearest_vals = cp.asnumpy(indexed_array_cupy[results_indices])


    return results_indices, results_nearest_vals


def take_closest_numpy(indexed_array, testing_values_array):
    indexed_array_numpy = np.array(indexed_array)
    testing_values_array_numpy = np.array(testing_values_array)

    indexed_array_numpy_tiled = np.tile(indexed_array_numpy, (testing_values_array_numpy.shape[0],1)) 

    dist_array = np.absolute(indexed_array_numpy_tiled.transpose()-testing_values_array_numpy).transpose()
    results_indices = dist_array.argmin(axis=1)
    results_nearest_vals = indexed_array_numpy[results_indices]


    return results_indices, results_nearest_vals


class delaunay_obj:
    def __init__(self, np_points, delaunay_tri_color, zslice1 = None, zslice2 = None):
        self.zslice1 = zslice1
        self.zslice2 = zslice2
        self.points_arr = np_points
        self.tricolor = delaunay_tri_color
        self.delaunay_triangulation = self.scipy_delaunay_triangulation(np_points)
        self.delaunay_line_set = None

    def scipy_delaunay_triangulation(self, numpy_points):
        delaunay_triang = scipy.spatial.Delaunay(numpy_points)
        return delaunay_triang

    def collect_edges(self, tri):
        edges = set()

        def sorted_tuple(a,b):
            return (a,b) if a < b else (b,a)
        # Add edges of tetrahedron (sorted so we don't add an edge twice, even if it comes in reverse order).
        for (i0, i1, i2, i3) in tri.simplices:
            edges.add(sorted_tuple(i0,i1))
            edges.add(sorted_tuple(i0,i2))
            edges.add(sorted_tuple(i0,i3))
            edges.add(sorted_tuple(i1,i2))
            edges.add(sorted_tuple(i1,i3))
            edges.add(sorted_tuple(i2,i3))
        return edges

    def generate_lineset(self):
        self.delaunay_line_set = self.line_set(self.points_arr, self.delaunay_triangulation, self.tricolor)

    def line_set(self, points, tri, color):
        edges = self.collect_edges(tri)
        colors = [[color[0], color[1], color[2]] for i in range(len(edges))]
        x = np.array([])
        y = np.array([])
        z = np.array([])
        for (i,j) in edges:
            x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
            y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
            z = np.append(z, [points[i, 2], points[j, 2], np.nan])

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(edges)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
    





def nearest_zslice_vals_and_indices_numpy_generic(zvals_list, 
                                                  test_arr_1d,
                                                max_array_NxN_size,
                                                progress_bar_level_task_obj):
    
    num_test_points = test_arr_1d.shape[0]
    num_zvals = len(zvals_list)

    nearest_zslice_index_array_final = np.empty(num_test_points, dtype = np.int64)
    nearest_zslice_vals_array_final = np.empty(num_test_points)
    
    
    if num_zvals*num_test_points > max_array_NxN_size:
        
        chunk_size = math.floor(max_array_NxN_size/num_zvals)
        num_point_chunks = math.ceil(num_test_points/chunk_size)
        next_pt_index = 0
        current_pt_index = 0

        task_description = "[cyan]Processing point chunks"
        chunk_task = progress_bar_level_task_obj.add_task(task_description, total=num_point_chunks)
        while next_pt_index < num_test_points:
            if num_test_points - current_pt_index > chunk_size:
                next_pt_index = current_pt_index + chunk_size
            else:
                next_pt_index = num_test_points

            nearest_zslice_index_array_intermediate, nearest_zslice_vals_intermediate = point_containment_tools.take_closest_numpy(zvals_list, test_arr_1d[current_pt_index:next_pt_index])

            nearest_zslice_index_array_final[current_pt_index:next_pt_index] = nearest_zslice_index_array_intermediate
            nearest_zslice_vals_array_final[current_pt_index:next_pt_index] = nearest_zslice_vals_intermediate

            del nearest_zslice_index_array_intermediate
            del nearest_zslice_vals_intermediate

            current_pt_index = next_pt_index

            progress_bar_level_task_obj.update(chunk_task, advance=1)
        progress_bar_level_task_obj.update(chunk_task, visible=False)

    else:
        nearest_zslice_index_array_final, nearest_zslice_vals_array_final = point_containment_tools.take_closest_numpy(zvals_list, test_arr_1d) 

    return nearest_zslice_index_array_final, nearest_zslice_vals_array_final
    
    
    

def nearest_zslice_vals_and_indices_cupy_generic(zvals_list, 
                                                test_arr_1d,
                                                max_array_NxN_size,
                                                progress_bar_level_task_obj):
    
    num_test_points = test_arr_1d.shape[0]
    num_zvals = len(zvals_list)

    nearest_zslice_index_array_final = np.empty(num_test_points, dtype = np.int64)
    nearest_zslice_vals_array_final = np.empty(num_test_points)
    
    
    if num_zvals*num_test_points > max_array_NxN_size:
        
        chunk_size = math.floor(max_array_NxN_size/num_zvals)
        num_point_chunks = math.ceil(num_test_points/chunk_size)
        next_pt_index = 0
        current_pt_index = 0

        task_description = "[cyan]Processing point chunks"
        chunk_task = progress_bar_level_task_obj.add_task(task_description, total=num_point_chunks)
        while next_pt_index < num_test_points:
            if num_test_points - current_pt_index > chunk_size:
                next_pt_index = current_pt_index + chunk_size
            else:
                next_pt_index = num_test_points

            nearest_zslice_index_array_intermediate, nearest_zslice_vals_intermediate = point_containment_tools.take_closest_cupy(zvals_list, test_arr_1d[current_pt_index:next_pt_index])

            nearest_zslice_index_array_final[current_pt_index:next_pt_index] = nearest_zslice_index_array_intermediate
            nearest_zslice_vals_array_final[current_pt_index:next_pt_index] = nearest_zslice_vals_intermediate

            del nearest_zslice_index_array_intermediate
            del nearest_zslice_vals_intermediate

            current_pt_index = next_pt_index

            progress_bar_level_task_obj.update(chunk_task, advance=1)
        progress_bar_level_task_obj.update(chunk_task, visible=False)

    else:
        nearest_zslice_index_array_final, nearest_zslice_vals_array_final = point_containment_tools.take_closest_cupy(zvals_list, test_arr_1d) 

    return nearest_zslice_index_array_final, nearest_zslice_vals_array_final