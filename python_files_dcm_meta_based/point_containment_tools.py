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
    delaunay_triangulation_obj_temp = delaunay_obj(adjacent_zslice_threeD_data_list, pcd_color)
    return delaunay_triangulation_obj_temp


def test_zslice_wise_containment_delaunay_parallel(parallel_pool, delaunay_obj_list, test_points_list):
    pool = parallel_pool
    for ind,pts in enumerate(test_pts_arr):
        #print(tri.find_simplex(pts) >= 0)  # True if point lies within poly)
        if delaunay_triangulation_obj.delaunay_triangulation.find_simplex(pts) >= 0:
            test_pt_colors[ind,:] = np.array([0,1,0]) # paint green
        else: 
            test_pt_colors[ind,:] = np.array([1,0,0]) # paint red



class delaunay_obj:
    def __init__(self, np_points, delaunay_tri_color):
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