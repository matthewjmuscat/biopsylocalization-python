import scipy
import numpy as np

def adjacent_slice_delaunay_parallel(parallel_pool, threeD_data_zlsice_list):
    """
    Feed raw data, not interpolated. Data should be a list whose entries are zslices (constant z)
    and each entry (zslice) in the list should be a numpy array of points
    """
    pool = parallel_pool
    adjacent_zslice_threeD_data_list = [[threeD_data_zlsice_list[j],threeD_data_zlsice_list[j+1]] for j in range(0, len(threeD_data_zlsice_list)-1)]
    delaunay_triangulation_obj_zslicewise_list = pool.map(adjacent_zlsice_delaunay_triangulation, adjacent_zslice_threeD_data_list)
    return delaunay_triangulation_obj_zslicewise_list

def adjacent_zlsice_delaunay_triangulation(adjacent_zslice_threeD_data_list):
    pcd_color = np.random.uniform(0, 0.7, size=3)
    zslice1 = adjacent_zslice_threeD_data_list[0][0,2]
    zslice2 = adjacent_zslice_threeD_data_list[1][0,2]
    threeDdata_array_adjacent_slices_arr = np.asarray(adjacent_zslice_threeD_data_list, dtype=float)
    delaunay_triangulation_obj_temp = delaunay_obj(threeDdata_array_adjacent_slices_arr, pcd_color, zslice1, zslice2)
    return delaunay_triangulation_obj_temp


def test_zslice_wise_containment_delaunay_parallel(parallel_pool, delaunay_obj_list, test_points_list):
    pool = parallel_pool
    test_points_arguments_list = [(delaunay_obj_list,test_points_list[j]) for j in range(len(test_points_list))]
    test_points_result = pool.starmap(zslice_wise_test_point_containment, test_points_arguments_list)
    return test_points_result
    
    
def zslice_wise_test_point_containment(delauney_object_list,test_point):
    pt_contained = False 
    for delaunay_obj_index, delaunay_obj in enumerate(delauney_object_list):
        #tri.find_simplex(pts) >= 0  is True if point lies within poly)
        if delaunay_obj.delaunay_triangulation.find_simplex(test_point) >= 0:
            zslice1 = delaunay_obj.zslice1
            zslice2 = delaunay_obj.zslice2
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
        delaunay_obj_index = None
    return pt_contained, zslice1, zslice2, delaunay_obj_contained_in_index, test_pt_color, test_point



class delaunay_obj:
    def __init__(self, np_points, delaunay_tri_color, zslice1 = None, zslice2 = None):
        self.zslice1 = zslice1
        self.zslice2 = zslice2
        self.delaunay_triangulation = self.scipy_delaunay_triangulation(np_points)
        self.delaunay_line_set = self.line_set(np_points, self.delaunay_triangulation, delaunay_tri_color)

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