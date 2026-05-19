import numpy as np
from shapely.geometry import Point, Polygon
import custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p
import anatomy_reconstructor_tools


class interpolation_information_obj:
    def __init__(self,num_z_slices_raw):
        self.interpolate_distance = None
        self.scipylinesegments_by_zslice_keys_dict = {}
        self.numpoints_after_interpolation_per_zslice_dict = {}
        self.numpoints_raw_per_zslice_dict = {}
        self.interpolated_pts_list = []
        self.interpolated_pts_np_arr = None
        self.num_z_slices_raw = num_z_slices_raw
        #self.z_slice_seg_obj_list_temp = None
        self.endcaps_points = []
        self.interpolated_pts_with_end_caps_list = None
        self.interpolated_pts_with_end_caps_np_arr = None 

    def serial_analyze(self,three_Ddata_list,interp_dist):
        self.interpolate_distance = interp_dist
        for threeDdata_zslice in three_Ddata_list:
            result = self.analyze_structure_slice(threeDdata_zslice)
            zslice_key = result[0] 
            z_slice_seg_obj_list = result[1]
            numpoints_raw_per_zslice = result[2]
            numpoints_after_interpolation_per_zslice_temp  = result[3]
            threeDdata_zslice_interpolated_list = result[4]

            self.scipylinesegments_by_zslice_keys_dict[zslice_key] = z_slice_seg_obj_list
            self.numpoints_raw_per_zslice_dict[zslice_key] = numpoints_raw_per_zslice
            self.numpoints_after_interpolation_per_zslice_dict[zslice_key] = numpoints_after_interpolation_per_zslice_temp
            
            threeDdata_zslice_interpolated_arr = np.asarray(threeDdata_zslice_interpolated_list)
            self.interpolated_pts_list.append(threeDdata_zslice_interpolated_arr)
        self.interpolated_pts_np_arr = np.vstack(self.interpolated_pts_list)


    
    def parallel_analyze(self, parallel_pool, three_Ddata_list,interp_dist):
        pool = parallel_pool
        self.interpolate_distance = interp_dist
        parallel_result = pool.map(self.analyze_structure_slice, three_Ddata_list)
        for result in parallel_result:
            zslice_key = result[0] 
            z_slice_seg_obj_list = result[1]
            numpoints_raw_per_zslice = result[2]
            numpoints_after_interpolation_per_zslice_temp  = result[3]
            threeDdata_zslice_interpolated_list = result[4]

            self.scipylinesegments_by_zslice_keys_dict[zslice_key] = z_slice_seg_obj_list
            self.numpoints_raw_per_zslice_dict[zslice_key] = numpoints_raw_per_zslice
            self.numpoints_after_interpolation_per_zslice_dict[zslice_key] = numpoints_after_interpolation_per_zslice_temp
            
            threeDdata_zslice_interpolated_arr = np.asarray(threeDdata_zslice_interpolated_list)
            self.interpolated_pts_list.append(threeDdata_zslice_interpolated_arr)
        self.interpolated_pts_np_arr = np.vstack(self.interpolated_pts_list)

                
    
    def analyze_structure_slice(self, threeDdata_zslice):
        interp_dist = self.interpolate_distance
        numpoints_raw_per_zslice_temp = None
        numpoints_after_interpolation_per_zslice_temp = None
        z_val = threeDdata_zslice[0,2] 
        current_zslice_num_points = np.size(threeDdata_zslice,0)
        #z_slice_seg_obj_list_temp = self.create_zslice(z_val, current_zslice_num_points)
        num_segments_in_zslice = current_zslice_num_points
        zslice_key = z_val
        z_slice_seg_obj_list_temp = [None]*num_segments_in_zslice
        #self.numpoints_raw_per_zslice_dict[zslice_key] = num_points_in_zslice_raw
        numpoints_raw_per_zslice_temp = current_zslice_num_points
        

        threeDdata_zslice_interpolated_list = []
        zslice_pt_counter = current_zslice_num_points
        for j in range(0,current_zslice_num_points):
            if j < current_zslice_num_points-1:
                segment_points = threeDdata_zslice[j:j+2,0:3]
            else:
                segment_points = np.empty([2,3], dtype = float)
                segment_points[0,0:3] = threeDdata_zslice[j,0:3]
                segment_points[1,0:3] = threeDdata_zslice[0,0:3]
            
            segment_vec = segment_points[1,:] - segment_points[0,:]
            segment_length = np.linalg.norm(segment_vec)
            segment_obj = anatomy_reconstructor_tools.line_segment_obj(segment_vec,segment_length,segment_points)
            z_slice_seg_obj_list_temp[j] = segment_obj
            num_interpolations_on_seg = int(np.floor(segment_length/interp_dist))
            t_vals_with_end_points = np.linspace(0, 1, num=num_interpolations_on_seg+2) # generate the t values to evaluate along the longest segment 
            t_vals_without_end_points = t_vals_with_end_points[1:-1]
            interpolated_segment_list = []
            for t_val in t_vals_without_end_points:
                new_point = np.empty([1,3],dtype=float)
                new_point = segment_obj.new_xyz_via_vector_travel(t_val)
                interpolated_segment_list.append(new_point)
            
            first_point = segment_points[0,:]
            threeDdata_zslice_interpolated_list.append(first_point)
            for interpolated_point in interpolated_segment_list:
                threeDdata_zslice_interpolated_list.append(interpolated_point)
            zslice_pt_counter = zslice_pt_counter + num_interpolations_on_seg

        #self.numpoints_after_interpolation_per_zslice_dict[z_val] = zslice_pt_counter
        numpoints_after_interpolation_per_zslice_temp = zslice_pt_counter
        #self.insert_zslice(z_val, z_slice_seg_obj_list_temp)
        #for interpolated_point in threeDdata_zslice_interpolated_list:
        #    interpolated_pts_list.append(interpolated_point)
        #self.interpolated_pts_np_arr = np.asarray(self.interpolated_pts_list)
        # plot slicewise for debugging ?
        #plotting_funcs.plot_point_clouds(self.interpolated_pts_np_arr, label='Unknown')
        return zslice_key, z_slice_seg_obj_list_temp, numpoints_raw_per_zslice_temp, numpoints_after_interpolation_per_zslice_temp, threeDdata_zslice_interpolated_list

    def create_zslice(self, zslice_key, num_points_in_zslice_raw): # call this first
        num_segments_in_zslice = num_points_in_zslice_raw
        z_slice_seg_obj_list_temp = self.prealloc_zslice_list(num_segments_in_zslice)
        self.numpoints_raw_per_zslice_dict[zslice_key] = num_points_in_zslice_raw
        return z_slice_seg_obj_list_temp

    def prealloc_zslice_list(self,num_segments_in_zslice): # this is automatically used by the class
        zslice_segments_list = [None]*num_segments_in_zslice
        return zslice_segments_list      
        
    def insert_zslice(self, zslice_key,z_slice_seg_obj_list): # then use this after all iterations are complete
        self.scipylinesegments_by_zslice_keys_dict[zslice_key] = z_slice_seg_obj_list


    def create_fill(self, threeDdata_zslice, maximum_point_distance):
        if self.interpolated_pts_with_end_caps_list == None:
            self.interpolated_pts_with_end_caps_list = self.interpolated_pts_list.copy()
        else:
            pass

        z_val = threeDdata_zslice[0,2]
        min_x, min_y = np.amin(threeDdata_zslice[:,0:2], axis=0)
        max_x, max_y = np.amax(threeDdata_zslice[:,0:2], axis=0)
        grid_spacing = maximum_point_distance/np.sqrt(2)
        fill_points_xy_grid_arr = np.mgrid[min_x-grid_spacing:max_x+grid_spacing:grid_spacing, min_y-grid_spacing:max_y+grid_spacing:grid_spacing].reshape(2, -1).T
        fill_points_xyz_grid_arr = np.empty((len(fill_points_xy_grid_arr),3), dtype = float)
        fill_points_xyz_grid_arr[:,0:2] = fill_points_xy_grid_arr
        fill_points_xyz_grid_arr[:,2] = z_val
        fill_points_xyz_grid_list = fill_points_xyz_grid_arr.tolist()
        twoD_zslice_data_arr = threeDdata_zslice[:,0:2]
        twoD_zslice_data_list = twoD_zslice_data_arr.tolist()
        zslice_polygon_shapely = Polygon(twoD_zslice_data_list)
        threeDdata_zslice_fill_list = []
        for index, test_point in enumerate(fill_points_xyz_grid_list):
            test_point_shapely = Point(test_point)
            if test_point_shapely.within(zslice_polygon_shapely):
                threeDdata_zslice_fill_list.append(test_point)
        for fill_point in threeDdata_zslice_fill_list:
            fill_point_as_arr = np.asarray(fill_point)
            self.endcaps_points.append(fill_point_as_arr)
            self.interpolated_pts_with_end_caps_list.append(fill_point_as_arr)
        self.interpolated_pts_with_end_caps_np_arr = np.vstack(self.interpolated_pts_with_end_caps_list)


    def create_fill_new(self, threeDdata_zslice, maximum_point_distance, 
                   kernel_type="one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized"):
        """
        Fill a given z-slice using the GPU-accelerated point-in-polygon method.
        Instead of appending points one by one, this version stores points in NumPy arrays
        using vectorized concatenation. The output (the stored points) remains identical
        so that downstream code can continue to work as expected.
        
        Parameters:
        threeDdata_zslice : numpy.ndarray
            An (N, 3) array representing the points (with constant z) that form the polygon.
        maximum_point_distance : float
            Parameter used to determine grid spacing as maximum_point_distance/√2.
        kernel_type : str
            The GPU kernel type to use.
        """
        # Initialize the storage arrays if needed.
        # Instead of lists, we now store points in NumPy arrays.
        if not hasattr(self, "endcaps_points_np") or self.endcaps_points_np is None:
            self.endcaps_points_np = np.empty((0, 3), dtype=float)
        if not hasattr(self, "interpolated_pts_with_end_caps_np_arr") or self.interpolated_pts_with_end_caps_np_arr is None:
            self.interpolated_pts_with_end_caps_np_arr = np.empty((0, 3), dtype=float)

        # Determine constant z value.
        z_val = threeDdata_zslice[0, 2]

        # Compute the 2D bounding box.
        min_x, min_y = np.amin(threeDdata_zslice[:, 0:2], axis=0)
        max_x, max_y = np.amax(threeDdata_zslice[:, 0:2], axis=0)

        # Calculate grid spacing.
        grid_spacing = maximum_point_distance / np.sqrt(2)

        # Create a grid of candidate (x, y) points.
        xx, yy = np.meshgrid(
            np.arange(min_x - grid_spacing, max_x + grid_spacing, grid_spacing),
            np.arange(min_y - grid_spacing, max_y + grid_spacing, grid_spacing)
        )
        candidate_xy = np.column_stack((xx.ravel(), yy.ravel()))
        candidate_points = np.column_stack((candidate_xy, np.full(candidate_xy.shape[0], z_val)))
        num_candidate_pts = candidate_points.shape[0]

        # Reshape candidates to (1, num_candidate_pts, 3) for the GPU kernel.
        candidates_all = candidate_points.reshape(1, num_candidate_pts, 3)

        # Prepare the polygon as a list with a single slice.
        all_structures_slices = [[threeDdata_zslice]]
        mapping_array = np.array([0], dtype=np.int32)

        # Run the GPU-based point-in-polygon test.
        containment_results_cp_arr, _ = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
            all_structures_slices,
            candidates_all,
            mapping_array,
            constant_z_slice_polygons_handler_option='auto-close-if-open',
            remove_consecutive_duplicate_points_in_polygons=True,
            log_sub_dirs_list=[],
            log_file_name=None,
            include_edges_in_log=False,
            kernel_type=kernel_type
        )

        # Fetch the boolean mask (assumed shape: (num_candidate_pts,)).
        result_mask = containment_results_cp_arr[0].get()

        # Select the candidate points that are inside the polygon.
        valid_points = candidate_points[result_mask]

        # Use vectorized concatenation to update the storage arrays.
        self.endcaps_points_np = np.concatenate((self.endcaps_points_np, valid_points), axis=0)
        self.interpolated_pts_with_end_caps_np_arr = np.concatenate(
            (self.interpolated_pts_with_end_caps_np_arr, valid_points), axis=0
        )

        # Optionally, if you need to preserve the old "list" interface for compatibility,
        # you can add properties that convert the arrays to lists on-the-fly.



    def create_fill_new_v2(self, threeDdata_zslice, maximum_point_distance, 
                      kernel_type="one_to_one_pip_kernel_advanced_reparameterized_version_gpu_memory_performance_optimized"):
        """
        Fill a given z-slice using the GPU-accelerated point-in-polygon method.
        Instead of appending points one by one, this version stores points in NumPy arrays
        using vectorized concatenation. It now preserves previously stored non-end cap points 
        from self.interpolated_pts_list to ensure the overall results remain identical.

        Parameters:
        threeDdata_zslice : numpy.ndarray
            An (N, 3) array representing the points (with constant z) that form the polygon.
        maximum_point_distance : float
            Parameter used to determine grid spacing as maximum_point_distance/√2.
        kernel_type : str
            The GPU kernel type to use.
        """
        # Initialize storage for new fill points (end caps).
        if not hasattr(self, "endcaps_points_np") or self.endcaps_points_np is None:
            self.endcaps_points_np = np.empty((0, 3), dtype=float)
        
        # Initialize the combined storage array.
        # If not already set, initialize it with the preexisting points from self.interpolated_pts_list.
        if (not hasattr(self, "interpolated_pts_with_end_caps_np_arr") or 
            self.interpolated_pts_with_end_caps_np_arr is None or 
            self.interpolated_pts_with_end_caps_np_arr.shape[0] == 0):
            if hasattr(self, "interpolated_pts_list") and self.interpolated_pts_list is not None and len(self.interpolated_pts_list) > 0:
                self.interpolated_pts_with_end_caps_np_arr = np.vstack(self.interpolated_pts_list)
            else:
                self.interpolated_pts_with_end_caps_np_arr = np.empty((0, 3), dtype=float)

        # Determine constant z value.
        z_val = threeDdata_zslice[0, 2]

        # Compute the 2D bounding box.
        min_x, min_y = np.amin(threeDdata_zslice[:, 0:2], axis=0)
        max_x, max_y = np.amax(threeDdata_zslice[:, 0:2], axis=0)

        # Calculate grid spacing.
        grid_spacing = maximum_point_distance / np.sqrt(2)

        # Create a grid of candidate (x, y) points.
        xx, yy = np.meshgrid(
            np.arange(min_x - grid_spacing, max_x + grid_spacing, grid_spacing),
            np.arange(min_y - grid_spacing, max_y + grid_spacing, grid_spacing)
        )
        candidate_xy = np.column_stack((xx.ravel(), yy.ravel()))
        candidate_points = np.column_stack((candidate_xy, np.full(candidate_xy.shape[0], z_val)))
        num_candidate_pts = candidate_points.shape[0]

        # Reshape candidates to (1, num_candidate_pts, 3) for the GPU kernel.
        candidates_all = candidate_points.reshape(1, num_candidate_pts, 3)

        # Prepare the polygon as a list with a single slice.
        all_structures_slices = [[threeDdata_zslice]]
        mapping_array = np.array([0], dtype=np.int32)

        # Run the GPU-based point-in-polygon test.
        containment_results_cp_arr, _ = custom_raw_kernel_cuda_cuspatial_one_to_one_p_in_p.custom_point_containment_mother_function(
            all_structures_slices,
            candidates_all,
            mapping_array,
            constant_z_slice_polygons_handler_option='auto-close-if-open',
            remove_consecutive_duplicate_points_in_polygons=True,
            log_sub_dirs_list=[],
            log_file_name=None,
            include_edges_in_log=False,
            kernel_type=kernel_type
        )

        # Fetch the boolean mask (assumed shape: (num_candidate_pts,)).
        result_mask = containment_results_cp_arr[0].get()

        # Select the candidate points that are inside the polygon.
        valid_points = candidate_points[result_mask]

        # Update the storage arrays with the new fill points.
        self.endcaps_points_np = np.concatenate((self.endcaps_points_np, valid_points), axis=0)
        self.interpolated_pts_with_end_caps_np_arr = np.concatenate(
            (self.interpolated_pts_with_end_caps_np_arr, valid_points), axis=0
        )