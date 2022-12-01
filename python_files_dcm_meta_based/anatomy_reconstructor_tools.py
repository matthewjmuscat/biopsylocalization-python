import numpy as np
import pandas as pd
import scipy

def inter_zslice_interpolator(threeDdata_zslice_list, max_interpolation_dist):
    # check if each slice has the same number of points
    
    
    max_num_points_on_zslice = np.shape(threeDdata_zslice_list[0])[0]
    max_pt_zslice_index = 0
    for index, zslice_array in enumerate(threeDdata_zslice_list):
        num_points_zslice_j = np.shape(zslice_array)[0]
        if num_points_zslice_j > max_num_points_on_zslice:
            max_num_points_on_zslice = num_points_zslice_j
            max_pt_zslice_index = index
        else:
            pass

    unequal_slices_indices_num_points_list = []
    for index, zslice_array in enumerate(threeDdata_zslice_list):
        num_points_zslice_j = np.shape(zslice_array)[0]
        if num_points_zslice_j == max_num_points_on_zslice:
            pass
        else:
            unequal_slice = [index, num_points_zslice_j]
            unequal_slices_indices_num_points_list.append(unequal_slice)
            
    if len(unequal_slices_indices_num_points_list) != 0:
        print('\n largest num points: ', max_num_points_on_zslice, ' is on index: ', max_pt_zslice_index, '\n')
        threeDdata_equal_pt_zslice_list = threeDdata_consistent_num_pts_zslice_completer(threeDdata_zslice_list, unequal_slices_indices_num_points_list, max_num_points_on_zslice, max_pt_zslice_index)
    else:
        threeDdata_equal_pt_zslice_list = threeDdata_zslice_list.copy()
    
    print('\n All slices have equal number of points!\n')
    zslices_index_pairings_dict = perform_distance_minimization(threeDdata_equal_pt_zslice_list)
    #print(zslices_index_pairings_dict)
    interslice_interpolation_information = slice_point_pairings_interpolator(max_interpolation_dist, zslices_index_pairings_dict, threeDdata_equal_pt_zslice_list)
    return interslice_interpolation_information, threeDdata_equal_pt_zslice_list


def threeDdata_consistent_num_pts_zslice_completer(threeDdata_zslice_list, unequal_slices_indices_num_points_list, max_num_points_on_zslice, max_pt_zslice_index):
    threeDdata_equal_pt_zslice_list = threeDdata_zslice_list.copy()
    for unequal_slice_info in unequal_slices_indices_num_points_list:
        unequal_zslice_index = unequal_slice_info[0]
        original_num_points_on_unequal_zslice = unequal_slice_info[1]
        num_points_to_add = max_num_points_on_zslice - original_num_points_on_unequal_zslice
        
        for k in range(0,num_points_to_add):
            current_zslice = threeDdata_equal_pt_zslice_list[unequal_zslice_index]
            current_zslice_num_points = np.shape(current_zslice)[0]
            current_zslice_list = current_zslice.tolist()
            z_val =  current_zslice[0,2]
            longest_segment_length = 0
            longest_segment_index = 0
            segment_obj = None
            for j in range(0,current_zslice_num_points):
                if j < current_zslice_num_points-1:
                    segment_points = current_zslice[j:j+2,0:3]
                else:
                    segment_points = np.empty([2,3], dtype = float)
                    segment_points[0,0:3] = current_zslice[j,0:3]
                    segment_points[1,0:3] = current_zslice[0,0:3]

                x = segment_points[0:2,0]
                y = segment_points[0:2,1]
                
                segment_length = np.linalg.norm(segment_points[0]-segment_points[1])
                if segment_length > longest_segment_length:
                    longest_segment_length = segment_length
                    longest_segment_index = j
                    f_scipy_seg = scipy.interpolate.interp1d(x, y)
                    xnew = (x[0]+x[1])/2
                else: 
                    pass

            ynew = f_scipy_seg(xnew)   # use interpolation function returned by `interp1d`
            interpolated_point = [xnew,ynew,z_val]
            current_zslice_list.insert(longest_segment_index+1, interpolated_point)
            threeDdata_equal_pt_zslice_list[unequal_zslice_index] = np.asarray(current_zslice_list, dtype = float)
    
    return threeDdata_equal_pt_zslice_list         


def perform_distance_minimization(threeDdata_zslice_list):
    zslices_index_pairings_dict = {}
    num_points_in_all_slices = np.shape(threeDdata_zslice_list[0])[0] # this value should be the same for every slice
    test_pairings_list_all = build_pairings_list(num_points_in_all_slices)
    for index in range(len(threeDdata_zslice_list)-1):
        test_pairings_SOSQdist_list_all = [None]*num_points_in_all_slices
        current_zslice = threeDdata_zslice_list[index]
        current_zslice_zval = threeDdata_zslice_list[index][0,2]
        next_zslice_zval = threeDdata_zslice_list[index+1][0,2]
        next_zslice = threeDdata_zslice_list[index+1]
        for test_pairing_list_index, test_pairing_list in enumerate(test_pairings_list_all):
            sq_distances_point_pairs_list = [None]*num_points_in_all_slices
            for pair_index, pair in enumerate(test_pairing_list):
                pt_on_current_zslice = current_zslice[pair[0]]
                pt_on_next_zslice = next_zslice[pair[1]]
                pair_distance = np.linalg.norm(pt_on_current_zslice-pt_on_next_zslice)
                sq_distances_point_pairs_list[pair_index] = pair_distance**2
            sum_of_sq_distance = sum(sq_distances_point_pairs_list)
            test_pairings_SOSQdist_list_all[test_pairing_list_index] = sum_of_sq_distance
        minimum_SOSQ_dist_index = pd.Series(test_pairings_SOSQdist_list_all).idxmin()
        zslices_index_pairings_dict[(current_zslice_zval,next_zslice_zval)] = test_pairings_list_all[minimum_SOSQ_dist_index]
    return zslices_index_pairings_dict

def build_pairings_list(num_points_in_all_slices):
    test_pairings_list_all = [None]*num_points_in_all_slices
    point_indices = [x for x in range(num_points_in_all_slices)]
    for i in range(num_points_in_all_slices):
        test_pairings_list = [None]*num_points_in_all_slices
        for j in range(num_points_in_all_slices):
            adjacent_index = (j+i) % num_points_in_all_slices
            pairing = (point_indices[j], point_indices[adjacent_index])
            test_pairings_list[j] = pairing
        test_pairings_list_all[i] = test_pairings_list
    return test_pairings_list_all

def slice_point_pairings_interpolator(interpolation_dist, zslices_index_pairings_dict, threeDdata_zslice_list):
    interslice_interpolation_information = interslice_interpolation_information_obj(threeDdata_zslice_list)
    interslice_interpolation_information.analyze_structure(threeDdata_zslice_list, zslices_index_pairings_dict, interpolation_dist)
    return interslice_interpolation_information


class interslice_interpolation_information_obj:
    def __init__(self, threeDdata_zslice_list):
        self.linesegments_dict_by_adjacent_zslice_keys_dict = {} 
        #self.numpoints_after_interpolation_per_zslice_dict = {}
        #self.numpoints_raw_per_zslice_dict = {}
        self.interpolated_pts_list = []
        self.interpolated_pts_np_arr = None
        self.num_z_slices_raw = len(threeDdata_zslice_list)
        self.z_slice_seg_obj_list_temp = None
        self.max_interp_distance = None
        self.zslices_index_pairings_dict = None

    def analyze_structure(self, threeDdata_zslice_list, zslices_index_pairings_dict, max_interp_dist):
        self.interpolated_pts_list = threeDdata_zslice_list.copy()
        self.zslices_index_pairings_dict = zslices_index_pairings_dict
        self.max_interp_distance = max_interp_dist
        insert_index = 1
        for current_slice_index in range(self.num_z_slices_raw-1): # exclude last slice
            threeDdata_current_slice_arr = threeDdata_zslice_list[current_slice_index]
            adjacent_index = (current_slice_index+1) 
            threeDdata_adjacent_slice_arr = threeDdata_zslice_list[adjacent_index]
            threeDdata_interpolated_bt_two_zslices_list = self.analyze_adjacent_structure_slices(threeDdata_current_slice_arr, threeDdata_adjacent_slice_arr, max_interp_dist)
            for interpolated_slice_index,z_slice_arr in enumerate(threeDdata_interpolated_bt_two_zslices_list):
                self.interpolated_pts_list.insert(insert_index,z_slice_arr)
                insert_index = insert_index + 1 
            insert_index = insert_index + 1



    def analyze_adjacent_structure_slices(self, threeDdata_current_slice_arr, threeDdata_adjacent_slice_arr, max_interp_dist):
        linesegments_by_point_pairings_keys_dict = {}
        z_val_current = threeDdata_current_slice_arr[0,2] 
        z_val_adjacent = threeDdata_adjacent_slice_arr[0,2]
        current_zslice_num_points = np.size(threeDdata_current_slice_arr,0)
        current_zslice_num_segments = current_zslice_num_points
        adjacent_slice_key = (z_val_current,z_val_adjacent)
        threeDdata_interpolated_zslices_list_temp = []
        zvals_key = (z_val_current,z_val_adjacent)
        current_and_adjacent_zslices_index_pairings = self.zslices_index_pairings_dict[zvals_key]
        caa_ind_p = current_and_adjacent_zslices_index_pairings
        total_num_interpolations_counter = 0
        #insert_index = 1
        longest_segment_index = None
        longest_segment_length = 0
        for j in range(0,current_zslice_num_segments):
            point_pairings_key = caa_ind_p[j]
            current_zslice_pt_index = point_pairings_key[0]
            adjacent_zslice_pt_index = point_pairings_key[1]
            pt_indices_key = (current_zslice_pt_index,adjacent_zslice_pt_index)
            segment_points = np.empty([2,3], dtype=float)
            segment_points[0,:] = threeDdata_current_slice_arr[current_zslice_pt_index]
            segment_points[1,:] = threeDdata_adjacent_slice_arr[adjacent_zslice_pt_index]
            segment_vec = segment_points[1,:] - segment_points[0,:]
            segment_length = np.linalg.norm(segment_vec)
            segment_obj = line_segment_obj(segment_vec,segment_length,segment_points,zvals_key,pt_indices_key)
            if segment_length > longest_segment_length:
                longest_segment_index = j
                longest_segment_length = segment_length
                segment_obj.longest_segment_in_adjacent_slices = True
            linesegments_by_point_pairings_keys_dict[point_pairings_key] = segment_obj
    
        self.linesegments_dict_by_adjacent_zslice_keys_dict[adjacent_slice_key] = linesegments_by_point_pairings_keys_dict

        new_z_slices_vals_list = []
        # analyze longest segment
        longest_segment_point_pairings_key = caa_ind_p[longest_segment_index]
        longest_segment_obj = linesegments_by_point_pairings_keys_dict[longest_segment_point_pairings_key]
        longest_segment_obj_length = longest_segment_obj.segment_length
        num_interpolations_on_longest_segment = int(np.floor(longest_segment_obj_length/max_interp_dist))
        t_vals_with_end_points = np.linspace(0, 1, num=num_interpolations_on_longest_segment+2) # generate the t values to evaluate along the longest segment 
        t_vals_without_end_points = t_vals_with_end_points[1:-1]
        for t_val_ind,t_val in enumerate(t_vals_without_end_points):
            new_z_slices_vals_list.append(longest_segment_obj.coordinate_val('z',t_val)) # calculate new z_vals
        # loop through all new z vals to be added
        
        for z_val in new_z_slices_vals_list:
            new_z_slice = np.empty([current_zslice_num_segments,3], dtype = float)
            # loop through all segments
            for j, (segment_key, segment_obj) in enumerate(self.linesegments_dict_by_adjacent_zslice_keys_dict[adjacent_slice_key].items()):
                new_point = np.empty([1,3],dtype=float)
                new_point_xy = segment_obj.new_xy_vals_from_z(z_val)
                new_point[0,0] = new_point_xy[0] # new x val
                new_point[0,1] = new_point_xy[1] # new y val
                new_point[0,2] = z_val
                new_z_slice[j,:] = new_point
            threeDdata_interpolated_zslices_list_temp.append(new_z_slice)
        
        return threeDdata_interpolated_zslices_list_temp
        


class line_segment_obj:
    def __init__(self,segment_vector,seg_length,seg_end_points,zvals_key=None,pt_indices_key=None):
        self.segment_zslices = zvals_key
        self.segment_pt_indices = pt_indices_key
        self.segment_vector = segment_vector
        self.segment_end_points = np.empty([2,3], dtype = float) 
        try:
            for j in range(0, np.size(seg_end_points,axis=0)):
                self.segment_end_points[j] = seg_end_points[j]
        except:
            print('incorrect size/type of line segment points array')
        self.segment_length = seg_length
        self.unit_segment_vector = segment_vector/seg_length
        self.num_interpolations_on_segment = None
        self.longest_segment_in_adjacent_slices = False
    def coordinate_val(self,coordinate,t):
        if isinstance(coordinate, str) == False:
            raise Exception('Coordinate argument must be a string dtype')
        elif coordinate == 'x':
            c = 0
        elif coordinate == 'y': 
            c = 1
        elif coordinate == 'z':
            c = 2
        vi=self.segment_end_points[0,:]
        vi_c=vi[c]
        line_vec = self.segment_vector
        line_vec_c = line_vec[c]
        c_val = vi_c+line_vec_c*t
        return c_val
    def new_xy_vals_from_z(self, z_val):
        vi=self.segment_end_points[0,:]
        vi_z=vi[2]
        line_vec = self.segment_vector
        line_vec_z = line_vec[2]
        t_val = (z_val - vi_z)/line_vec_z
        x_val = self.coordinate_val('x',t_val)
        y_val = self.coordinate_val('y',t_val)
        return x_val, y_val
    def new_xyz_via_vector_travel(self, t_val):
        vi = self.segment_end_points[0,:]
        travel_vec = self.segment_vector
        vf = vi + t_val*travel_vec
        return vf 
