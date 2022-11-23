import numpy as np
import pandas as pd


def intra_zslice_interpolator(threeDdata_zslice_list):
    # check if each slice has the same number of points
    num_points_zslice0 = np.shape(threeDdata_zslice_list[0])[0]
    
    unequal_slices_indices_num_points_list = []
    for index, zslice_array in enumerate(threeDdata_zslice_list):
        num_points_zslice_j = np.shape(zslice_array)[0]
        if num_points_zslice_j == num_points_zslice0:
            pass
        else:
            unequal_slice = [index, num_points_zslice_j]
            unequal_slices_indices_num_points_list.append(unequal_slice)
            
    largest_num_points_zslice = num_points_zslice0
    largest_points_zslice_index = 0
    if len(unequal_slices_indices_num_points_list) != 0:
        for unequal_slice in unequal_slices_indices_num_points_list:
            if largest_num_points_zslice < unequal_slice[1]:
                largest_num_points_zslice = unequal_slice[1]
                largest_points_zslice_index = unequal_slice[0]
            else:
                pass
        print('\n largest num points: ', largest_num_points_zslice, ' is on index: ', largest_points_zslice_index, '\n')
    else:
        print('\n All slices have equal number of points!\n')
        zslices_index_pairings_dict = perform_distance_minimization(threeDdata_zslice_list)
        print(zslices_index_pairings_dict)
    return

def perform_distance_minimization(threeDdata_zslice_list):
    zslices_index_pairings_dict = {}
    num_points_in_all_slices = np.shape(threeDdata_zslice_list[0])[0]
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



