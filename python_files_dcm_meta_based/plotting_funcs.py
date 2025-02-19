from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import gc
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import json
import plotly.express as px
import point_containment_tools
import copy
import plotly.graph_objects as go
import misc_tools


def threeD_scatter_plotter(x,y,z):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return fig


def arb_threeD_scatter_plotter(*data_and_color,**text):
    """
    accepts arbitrary number of data to plot
    """
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for data in data_and_color:
        ax.scatter(data[0], data[1], data[2], c=data[3], marker=data[4]) 
        
    iterator = 1
    info_to_print = [x for x in text.items() if type(x[1])==str]
    for key, value in info_to_print:
        x_pos=1.15
        y_pos=0.5-0.05*iterator
        ax.text2D(x_pos, y_pos, "%s: %s" % (key, value), transform=ax.transAxes)
        iterator = iterator + 1
    set_axes_equal(ax)

    return fig

def arb_threeD_scatter_plotter_list(*data_and_color, **text):
    """
    accepts nested list of data
    """
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for data in data_and_color:
        ax.scatter(data[0], data[1], data[2], color=data[3], marker=data[4])
        zslice_prev = data[2][0]
        j=1
        for i in range(len(data[0])):
            zslice = data[2][i]
            if zslice == zslice_prev:
                ax.text(data[0][i],data[1][i],data[2][i],  '%s' % (str(j)), size=20, zorder=1, color='k')
                j=j+1
            else:
                j=1
                ax.text(data[0][i],data[1][i],data[2][i],  '%s' % (str(j)), size=20, zorder=1, color='k')
                j=j+1
            zslice_prev = zslice

    iterator = 1
    info_to_print = [x for x in text.items() if type(x[1])==str]
    for key, value in info_to_print:
        x_pos=1.15
        y_pos=0.5-0.05*iterator
        ax.text2D(x_pos, y_pos, "%s: %s" % (key, value), transform=ax.transAxes)
        iterator = iterator + 1
    set_axes_equal(ax)

    return fig

def arb_threeD_scatter_plotter_global(data_and_color):
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for data in data_and_color:
        ax.scatter(data[0], data[1], data[2], color=data[3], marker=data[4])

    set_axes_equal(ax)

    return fig

def add_line(figure,line):
    ax = figure.get_axes()
    ax[0].plot(line[:, 0], line[:, 1], line[:, 2], 'g')
    return figure


def plot_general_per_patient(per_patient_master_dict, structs_referenced_list, BX_plot_attr = ["raw","cen","cbfl","cbfls","rcBX"], OAR_plot_attr = ["raw"], DIL_plot_attr = ["raw"], NN = "DIL", **text):
    """
    accepts master data dict, if any plot attribute lists are empty nothing of that structure will be plotted
    """

    BX_plot_attr_copy = BX_plot_attr.copy()
    OAR_plot_attr_copy = OAR_plot_attr.copy()
    DIL_plot_attr_copy = DIL_plot_attr.copy()

    
    structs_referenced_list_subset = []
    if bool(BX_plot_attr):
        structs_referenced_list_subset.append(structs_referenced_list[0])
    if bool(OAR_plot_attr):
        structs_referenced_list_subset.append(structs_referenced_list[1])
    if bool(DIL_plot_attr):
        structs_referenced_list_subset.append(structs_referenced_list[2])


    translator = {"raw": "Raw contour pts","cen": "Structure centroid pts","cbfl": "Best fit line of centroid pts","cbfls": "Centroid line sample pts","rcBX": "Reconstructed structure pts","NN": "Nearest neighbours objects"}
    

    attributes_list_of_lists = [BX_plot_attr_copy,OAR_plot_attr_copy,DIL_plot_attr_copy]
    for att_LOL_index,attribute_list in enumerate(attributes_list_of_lists):
        for index,attr in enumerate(attribute_list):
            attributes_list_of_lists[att_LOL_index][index] = translator[attr]
         


    data_and_plotatts_list = []
    pydicom_item = per_patient_master_dict
    for struct_type in structs_referenced_list_subset:
        if struct_type == structs_referenced_list[0]:
            keys_to_plot = attributes_list_of_lists[0]
        if struct_type == structs_referenced_list[1]:
            keys_to_plot = attributes_list_of_lists[1]
        if struct_type == structs_referenced_list[2]:
            keys_to_plot = attributes_list_of_lists[2]
        for specific_structure in pydicom_item[struct_type]:
            for key in keys_to_plot:
                np_array_tranpose_list_and_plotatts = specific_structure[key].T.tolist()
                color_and_marker = plot_attributes(struct_type,key)
                np_array_tranpose_list_and_plotatts.append(color_and_marker[0])
                np_array_tranpose_list_and_plotatts.append(color_and_marker[1])
                np_array_tranpose_list_and_plotatts.append(color_and_marker[2])
                data_and_plotatts_list.append(np_array_tranpose_list_and_plotatts)




    # if NN is given as an empty string, then nearest neighbours wont be plotted
    if bool(NN):
        struct_type_BX = structs_referenced_list[0]
        color_and_marker = plot_attributes(struct_type_BX,translator["NN"])
        for specific_structure in pydicom_item[struct_type_BX]:
            if NN == 'all':
                NN_object_list = specific_structure[translator["NN"]]
            elif NN == 'DIL':
                NN_object_list = [x for x in specific_structure[translator["NN"]] if x.comparison_structure_type == structs_referenced_list[2]]
            elif NN == 'DILOAR':
                NN_object_list = [x for x in specific_structure[translator["NN"]] if x.comparison_structure_type == structs_referenced_list[2] or x.comparison_structure_type == structs_referenced_list[1]]
            for NN_object_parent in NN_object_list:
                for NN_object_child in NN_object_parent.NN_data_list:
                    shortest_dist_line_data_list_and_plotatts = np.array([NN_object_child.queried_BX_pt,NN_object_child.NN_pt_on_comparison_struct]).T.tolist()
                    shortest_dist_line_data_list_and_plotatts.append(color_and_marker[0])
                    shortest_dist_line_data_list_and_plotatts.append(color_and_marker[1])
                    shortest_dist_line_data_list_and_plotatts.append(color_and_marker[2])
                    data_and_plotatts_list.append(shortest_dist_line_data_list_and_plotatts)

                    
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    for data in data_and_plotatts_list:
        if data[5] == 'points':
            ax.scatter(data[0], data[1], data[2], color=data[3], marker=data[4])
        if data[5] == 'line':
            ax.plot(data[0], data[1], data[2], color=data[3])

    iterator = 1
    info_to_print = [x for x in text.items() if type(x[1])==str]
    for key, value in info_to_print:
        x_pos=1.15
        y_pos=0.5-0.05*iterator
        ax.text2D(x_pos, y_pos, "%s: %s" % (key, value), transform=ax.transAxes)
        iterator = iterator + 1
    set_axes_equal(ax)

    return fig


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_attributes(structure_type,data_type):
    marker_dict = {"Raw contour pts":'o',"Structure centroid pts":'x',"Centroid line sample pts":'+',"Reconstructed structure pts": '2',"Nearest neighbours objects": '*',"Best fit line of centroid pts": 'None'}
    color_dict = {"Bx ref":'r',"OAR ref":'b',"DIL ref":'m'}
    plot_dict = {"Raw contour pts":'points',"Structure centroid pts":'points',"Centroid line sample pts":'points',"Reconstructed structure pts": 'points',"Nearest neighbours objects": 'line',"Best fit line of centroid pts": 'line'}
    return [color_dict[structure_type],marker_dict[data_type],plot_dict[data_type]]





# function used for plotting delaunay triangulization in open3d (can also be used for plotting in matplotlib)
def collect_edges(tri):
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

# function used for plotting delaunay triangulization in open3d (a similar function was defined in a test file to plot in matplotlib, open3d is better)
def plot_tri_more_efficient_open3d(points, tri):
    edges = collect_edges(tri)
    colors = [[1, 0, 0] for i in range(len(edges))]
    x = np.array([])
    y = np.array([])
    z = np.array([])
    for (i,j) in edges:
        x = np.append(x, [points[i, 0], points[j, 0], np.nan])      
        y = np.append(y, [points[i, 1], points[j, 1], np.nan])      
        z = np.append(z, [points[i, 2], points[j, 2], np.nan])

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(edges)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    #o3d.visualization.draw_geometries([line_set,point_cloud])
    return line_set

def plot_point_clouds(*points_arr, label='Unknown'):
    geometry_list = []
    for points in points_arr:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        pcd_color = np.random.uniform(0, 0.7, size=3)
        point_cloud.paint_uniform_color(pcd_color)
        geometry_list.append(point_cloud)
    o3d.visualization.draw_geometries(geometry_list)

def plot_geometries(*geometries, label='Unknown', lookat_inp=None, up_inp=None, front_inp=None, zoom_inp=None, show_axes=False, axes_length=1.0, axes_origin=[0, 0, 0]):
    geom_list = []
    for geom_item in geometries:
        geom_list.append(geom_item)
    
    if show_axes:
        # Create axes lines
        axes = o3d.geometry.LineSet()
        origin = np.array(axes_origin)
        points = np.array([
            origin, origin + np.array([axes_length, 0, 0]),  # X-axis
            origin, origin + np.array([0, axes_length, 0]),  # Y-axis
            origin, origin + np.array([0, 0, axes_length])   # Z-axis
        ])
        lines = [
            [0, 1],  # X-axis
            [2, 3],  # Y-axis
            [4, 5]   # Z-axis
        ]
        colors = [
            [1, 0, 0],  # Red for X-axis
            [0, 1, 0],  # Green for Y-axis
            [0, 0, 1]   # Blue for Z-axis
        ]
        axes.points = o3d.utility.Vector3dVector(points.reshape(6,3))
        axes.lines = o3d.utility.Vector2iVector(lines)
        axes.colors = o3d.utility.Vector3dVector(colors)
        
        # Add axes to the list of geometries
        geom_list.append(axes)
    
    # Plot the geometries
    o3d.visualization.draw_geometries(geom_list)

def plot_geometries_with_axes(*geometries, label='Unknown'):
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    for geometry in geometries:
        viewer.add_geometry(geometry)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    

def plot_tri_immediately_efficient(points, line_set, *other_geometries, label='Unknown'):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    point_cloud.paint_uniform_color(pcd_color)
    geometry_list = [line_set,point_cloud]
    if len(other_geometries) == 0:
        o3d.visualization.draw_geometries(geometry_list)
    else:
        for i in other_geometries: 
            geometry_list.append(i)
        o3d.visualization.draw_geometries(geometry_list)
        #o3d.visualization.gui.Label3D(label, point_cloud.get_max_bound()+np.array([3,3,3]))

def plot_tri_immediately_efficient_multilineset(structure_points_arr, test_pointcloud, delaunay_objs_list, label='Unknown'):
    structure_point_cloud = o3d.geometry.PointCloud()
    structure_point_cloud.points = o3d.utility.Vector3dVector(structure_points_arr)
    pcd_color = np.random.uniform(0, 0.7, size=3)
    structure_point_cloud.paint_uniform_color(pcd_color)
    geometry_list = [structure_point_cloud, test_pointcloud]
    for delaunay_obj in delaunay_objs_list:
        line_set = delaunay_obj.delaunay_line_set
        geometry_list.append(line_set)
    
    o3d.visualization.draw_geometries(geometry_list)
    



def point_cloud_with_order_labels(points):
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(points)
    gui_instance = gui.Application.instance
    gui_instance.initialize()
    window = gui_instance.create_window("Mesh-Viewer", 1024, 750)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    scene.scene.add_geometry("mesh_name", pointcloud, rendering.MaterialRecord())
    bounds = pointcloud.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())
    zslice_prev = np.asarray(pointcloud.points)[0,2]
    j=1
    for i in range(np.shape(np.asarray(pointcloud.points))[0]):
        zslice = np.asarray(pointcloud.points)[i,2]
        if zslice == zslice_prev:
            scene.add_3d_label(np.asarray(pointcloud.points)[i,:], '%s' % (str(j)))
            j=j+1
        else:
            j=1
            scene.add_3d_label(np.asarray(pointcloud.points)[i,:], '%s' % (str(j)))
            j=j+1
        zslice_prev = zslice

    
    gui_instance.run()  # Run until user closes window
    gui_instance.quit()
    window.close()
    del window
    del gui_instance
    


def plot_two_point_clouds_side_by_side(points_arr_1, points_arr_2):
    point_cloud_1 = o3d.geometry.PointCloud()
    point_cloud_1.points = o3d.utility.Vector3dVector(points_arr_1)
    pcd_color_1 = np.array([0,1,0], dtype= float)
    point_cloud_1.paint_uniform_color(pcd_color_1)

    point_cloud_2 = o3d.geometry.PointCloud()
    point_cloud_2.points = o3d.utility.Vector3dVector(points_arr_2)
    pcd_color_2 = np.array([1,0,0], dtype= float)
    point_cloud_2.paint_uniform_color(pcd_color_2)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=100)
    vis.add_geometry(point_cloud_1)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=100)
    vis2.add_geometry(point_cloud_2)

    while True:
        vis.update_geometry(point_cloud_1)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(point_cloud_2)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()




def plot_point_cloud_and_trimesh_side_by_side(points_arr_1, tri_mesh):
    point_cloud_1 = o3d.geometry.PointCloud()
    point_cloud_1.points = o3d.utility.Vector3dVector(points_arr_1)
    pcd_color_1 = np.array([0,1,0], dtype= float)
    point_cloud_1.paint_uniform_color(pcd_color_1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='TopLeft', width=960, height=540, left=0, top=100)
    vis.add_geometry(point_cloud_1)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name='TopRight', width=960, height=540, left=960, top=100)
    vis2.add_geometry(tri_mesh)

    while True:
        vis.update_geometry(point_cloud_1)
        if not vis.poll_events():
            break
        vis.update_renderer()

        vis2.update_geometry(tri_mesh)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

    vis.destroy_window()
    vis2.destroy_window()



def create_dose_point_cloud(data_3d_arr, color_flattening_degree, paint_dose_color = True):
    point_cloud = o3d.geometry.PointCloud()
    data_all_slices_2d_arr = np.reshape(data_3d_arr, (-1,7))
    num_points = data_all_slices_2d_arr.shape[0]
    position_data_all_slices_2d_arr = data_all_slices_2d_arr[:,3:6]
    dose_data_all_slices_2d_arr = data_all_slices_2d_arr[:,6]
    max_dose = np.amax(dose_data_all_slices_2d_arr)
    min_dose = np.amin(dose_data_all_slices_2d_arr)
    point_cloud.points = o3d.utility.Vector3dVector(position_data_all_slices_2d_arr)
    if paint_dose_color == True:
        root = 1/color_flattening_degree
        pcd_color_arr = np.zeros((num_points,3))
        dose_data_all_slices_2d_arr_nth_rooted = np.power(dose_data_all_slices_2d_arr,root)
        max_dose_nth_rooted = np.power(max_dose,root)
        pcd_color_arr[:,0] = dose_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr[:,2] = dose_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr = pcd_color_arr/max_dose_nth_rooted
        pcd_color_arr[:,2] = 1 - pcd_color_arr[:,2]
        #pcd_color_arr[pcd_color_arr<0.1] = 1. # set all points less than a threshold to be white
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color_arr)
    else:
        pcd_color = np.array([0,0,0]) # paint everything black
        point_cloud.paint_uniform_color(pcd_color)

    return point_cloud

def create_dose_point_cloud_with_gradients(
    data_with_gradients_3d_arr,
    paint_dose_color=True,
    arrow_scale=1.0,
    truncate_below_dose=None,
    truncate_below_gradient_norm=None
):
    """
    Create a dose point cloud and gradient arrows using normalized gradient vectors and gradient norms.

    Parameters:
        data_with_gradients_3d_arr (numpy.ndarray): Array containing dose and gradient data (normalized gradients included).
        paint_dose_color (bool): Whether to paint the dose values as colors.
        arrow_scale (float): Scaling factor for gradient arrows.
        truncate_below_dose (float): Truncate points with dose values below this threshold.
        truncate_below_gradient_norm (float): Truncate gradient arrows with gradient norms below this threshold.

    Returns:
        tuple: Point cloud and arrow set for visualization.
    """


    # Extract data from input array
    point_cloud = o3d.geometry.PointCloud()
    data_all_slices_2d_arr = np.reshape(data_with_gradients_3d_arr, (-1, 14))
    
    position_data = data_all_slices_2d_arr[:, 3:6]  # x, y, z
    dose_data = data_all_slices_2d_arr[:, 6]       # dose
    gradient_norms = data_all_slices_2d_arr[:, 10]  # Gradient norm (|G|)
    gradient_vectors = data_all_slices_2d_arr[:, 11:14]  # Normalized Gx, Gy, Gz

    # Truncate based on dose values
    if truncate_below_dose is not None:
        dose_mask = dose_data >= truncate_below_dose
        position_data = position_data[dose_mask]
        dose_data = dose_data[dose_mask]
        gradient_norms = gradient_norms[dose_mask]
        gradient_vectors = gradient_vectors[dose_mask]

    # Truncate based on gradient norms
    if truncate_below_gradient_norm is not None:
        gradient_mask = gradient_norms >= truncate_below_gradient_norm
        position_data = position_data[gradient_mask]
        dose_data = dose_data[gradient_mask]
        gradient_norms = gradient_norms[gradient_mask]
        gradient_vectors = gradient_vectors[gradient_mask]

    # Apply arrow scaling factor
    gradient_vectors_scaled = gradient_vectors * arrow_scale

    # Create point cloud
    point_cloud.points = o3d.utility.Vector3dVector(position_data)
    if paint_dose_color:
        dose_data_log = np.log(dose_data + 1e-9)  # Avoid log(0)
        dose_data_log_normalized = (dose_data_log - np.min(dose_data_log)) / (
            np.max(dose_data_log) - np.min(dose_data_log) + 1e-9
        )
        pcd_color_arr = np.zeros((position_data.shape[0], 3))
        pcd_color_arr[:, 0] = dose_data_log_normalized  # Red for high log-dose
        pcd_color_arr[:, 2] = 1 - dose_data_log_normalized  # Blue for low log-dose
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color_arr)
    else:
        point_cloud.paint_uniform_color([0, 0, 0])  # Paint everything black

    # Create gradient arrows
    lines = []
    arrow_points = []
    arrow_colors = []
    
    gradient_norms_log = np.log(gradient_norms + 1e-9)  # Avoid log(0)
    gradient_norms_log_normalized = (gradient_norms_log - np.min(gradient_norms_log)) / (
        np.max(gradient_norms_log) - np.min(gradient_norms_log) + 1e-9
    )
    
    for i, point in enumerate(position_data):
        arrow_start = point
        arrow_end = point + gradient_vectors_scaled[i]
        arrow_points.append(arrow_start)
        arrow_points.append(arrow_end)
        lines.append([2 * i, 2 * i + 1])  # Each arrow is a pair of start and end points
        
        # Gradient norm-based coloring (e.g., red for high norm, blue for low)
        norm_color = [gradient_norms_log_normalized[i], 0, 1 - gradient_norms_log_normalized[i]]
        arrow_colors.append(norm_color)  # One color per line

    arrow_set = o3d.geometry.LineSet()
    arrow_set.points = o3d.utility.Vector3dVector(np.array(arrow_points))
    arrow_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    arrow_set.colors = o3d.utility.Vector3dVector(np.array(arrow_colors))
    
    return point_cloud, arrow_set







def create_thresholded_dose_point_cloud(data_3d_arr, color_flattening_degree, paint_dose_color = True, lower_bound_percent = 10):
    point_cloud = o3d.geometry.PointCloud()
    data_all_slices_2d_arr = np.reshape(data_3d_arr, (-1,7))
    dose_data_all_slices_2d_arr = data_all_slices_2d_arr[:,6]
    max_dose_og = np.amax(dose_data_all_slices_2d_arr)
    data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr[data_all_slices_2d_arr[:, 6]  > float(max_dose_og*lower_bound_percent/100)]
    num_points = data_all_slices_2d_arr_thresholded.shape[0]
    position_data_all_slices_2d_arr = data_all_slices_2d_arr_thresholded[:,3:6]
    dose_data_all_slices_2d_arr = data_all_slices_2d_arr_thresholded[:,6]
    min_dose = np.amin(dose_data_all_slices_2d_arr)
    point_cloud.points = o3d.utility.Vector3dVector(position_data_all_slices_2d_arr)
    if paint_dose_color == True:
        root = 1/color_flattening_degree
        pcd_color_arr = np.zeros((num_points,3))
        dose_data_all_slices_2d_arr_nth_rooted = np.power(dose_data_all_slices_2d_arr,root)
        max_dose_og_nth_rooted = np.power(max_dose_og,root)
        pcd_color_arr[:,0] = dose_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr[:,2] = dose_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr = pcd_color_arr/max_dose_og_nth_rooted
        pcd_color_arr[:,2] = 1 - pcd_color_arr[:,2]
        #pcd_color_arr[pcd_color_arr<0.1] = 1. # set all points less than a threshold to be white
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color_arr)
    else:
        pcd_color = np.array([0,0,0]) # paint everything black
        point_cloud.paint_uniform_color(pcd_color)

    return point_cloud





def dose_point_cloud_with_dose_labels(data_3d_arr, paint_dose_with_color = True):
    pointcloud = create_dose_point_cloud(data_3d_arr, paint_dose_color = paint_dose_with_color)
    gui_instance = gui.Application.instance
    gui_instance.initialize()
    window = gui_instance.create_window("Mesh-Viewer", 1024, 750)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)
    scene.scene.add_geometry("mesh_name", pointcloud, rendering.MaterialRecord())
    bounds = pointcloud.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())
    
    data_all_slices_2d_arr = np.reshape(data_3d_arr, (-1,7))
    num_points = data_all_slices_2d_arr.shape[0]
    position_data_all_slices_2d_arr = data_all_slices_2d_arr[:,3:6]
    dose_data_all_slices_2d_arr = data_all_slices_2d_arr[:,6]
    for i in range(num_points):
        scene.add_3d_label(position_data_all_slices_2d_arr[i,:], '{dose_val}'.format(dose_val = round(dose_data_all_slices_2d_arr[i],1)))

    gui_instance.run()  # Run until user closes window
    gui_instance.quit()
    window.close()
    del window
    del gui_instance



def dose_point_cloud_with_dose_labels_for_animation(NN_pts_on_dose_lattice_arr, NN_doses_on_dose_lattice_arr, queried_bx_pts_arr, queried_bx_pts_assigned_doses_arr, num_dose_NN_per_bx_pt, draw_lines = True):
    NN_doses_locations_pointcloud = point_containment_tools.create_point_cloud(NN_pts_on_dose_lattice_arr)
    queried_bx_pts_locations_pointcloud = point_containment_tools.create_point_cloud(queried_bx_pts_arr, color = np.array([0,1,0]))
    
    gui_instance = gui.Application.instance
    gui_instance.initialize()
    window = gui_instance.create_window("Mesh-Viewer", 1024, 750)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    scene.scene.add_geometry("Dose pcd", NN_doses_locations_pointcloud, rendering.MaterialRecord())
    scene.scene.add_geometry("Queried bx pts pcd", queried_bx_pts_locations_pointcloud, rendering.MaterialRecord())
    bounds = NN_doses_locations_pointcloud.get_axis_aligned_bounding_box()
    scene.setup_camera(60, bounds, bounds.get_center())
    
    num_bx_points = queried_bx_pts_arr.shape[0]
    num_NN_dose_points = NN_pts_on_dose_lattice_arr.shape[0]

    line_set_list = []
    for i in range(num_bx_points):
        scene.add_3d_label(queried_bx_pts_arr[i,:], '{dose_val}'.format(dose_val = round(queried_bx_pts_assigned_doses_arr[i],1)))
        if draw_lines == True:
            line_set_points_bx_pt_then_NN_arr = np.concatenate((np.expand_dims(queried_bx_pts_arr[i,:], axis=0),NN_pts_on_dose_lattice_arr[i*num_dose_NN_per_bx_pt:i*num_dose_NN_per_bx_pt+num_dose_NN_per_bx_pt,:]), axis=0)
            line_set_lines = [[0,j+1] for j in range(num_dose_NN_per_bx_pt)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_set_points_bx_pt_then_NN_arr)
            line_set.lines = o3d.utility.Vector2iVector(line_set_lines)
            line_set_list.append(copy.copy(line_set))
    
    line_mat = o3d.visualization.rendering.MaterialRecord()
    line_mat.shader = "unlitLine"
    line_mat.line_width = 10
    for line_set in line_set_list:
        scene.scene.add_geometry("", line_set, line_mat)
    
    for i in range(num_NN_dose_points):
        scene.add_3d_label(NN_pts_on_dose_lattice_arr[i,:], '{dose_val}'.format(dose_val = round(NN_doses_on_dose_lattice_arr[i],1)))

    gui_instance.run()  # Run until user closes window
    gui_instance.quit()
    window.close()
    del window
    del gui_instance



def dose_point_cloud_with_lines_only_for_animation(org_sampled_bx_pcd, thresholded_dose_pcd, NN_pts_on_dose_lattice_arr, queried_bx_pts_arr, num_dose_NN_per_bx_pt, draw_lines = True):
    NN_doses_locations_pointcloud = point_containment_tools.create_point_cloud(NN_pts_on_dose_lattice_arr)
    queried_bx_pts_locations_pointcloud = point_containment_tools.create_point_cloud(queried_bx_pts_arr, color = np.array([0,1,0]))
    
    geometry_list = [org_sampled_bx_pcd, thresholded_dose_pcd, NN_doses_locations_pointcloud,queried_bx_pts_locations_pointcloud]
    
    num_bx_points = queried_bx_pts_arr.shape[0]
    line_color = np.array([0,1,1])
    if draw_lines == True:
        line_set_list = []
        for i in range(num_bx_points):
            line_set_points_bx_pt_then_NN_arr = np.concatenate((np.expand_dims(queried_bx_pts_arr[i,:], axis=0),NN_pts_on_dose_lattice_arr[i*num_dose_NN_per_bx_pt:i*num_dose_NN_per_bx_pt+num_dose_NN_per_bx_pt,:]), axis=0)
            line_set_lines = [[0,j+1] for j in range(num_dose_NN_per_bx_pt)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line_set_points_bx_pt_then_NN_arr)
            line_set.lines = o3d.utility.Vector2iVector(line_set_lines)
            line_set.paint_uniform_color(line_color)
            line_set_list.append(copy.copy(line_set))
    
        geometry_list = geometry_list + line_set_list
    
    o3d.visualization.draw_geometries(geometry_list)
    return geometry_list







def plot_two_views_side_by_side(points_pcd_1_list, view_1_json_path, points_pcd_2_list, view_2_json_path):
    view_1_json_path = str(view_1_json_path)
    view_2_json_path = str(view_2_json_path)
    
    vis_1 = o3d.visualization.Visualizer()
    with open(view_1_json_path) as json_file_1:
        params_dict_1 = json.load(json_file_1)
        height_1 = params_dict_1["intrinsic"]["height"]
        width_1 = params_dict_1["intrinsic"]["width"]
    vis_1.create_window(window_name='TopLeft', width=width_1, height=height_1, left=0, top=100)
    for pcd_1_item in points_pcd_1_list:
        vis_1.add_geometry(pcd_1_item)
    ctr_1 = vis_1.get_view_control()
    parameters_1 = o3d.io.read_pinhole_camera_parameters(view_1_json_path)
    ctr_1.convert_from_pinhole_camera_parameters(parameters_1)

    vis_2 = o3d.visualization.Visualizer()
    with open(view_2_json_path) as json_file_2:
        params_dict_2 = json.load(json_file_2)
        height_2 = params_dict_2["intrinsic"]["height"]
        width_2 = params_dict_2["intrinsic"]["width"]
    vis_2.create_window(window_name='TopRight', width=width_2, height=height_2, left=960, top=100)
    for pcd_2_item in points_pcd_2_list:
        vis_2.add_geometry(pcd_2_item)
    ctr_2 = vis_2.get_view_control()
    parameters_2 = o3d.io.read_pinhole_camera_parameters(view_2_json_path)
    ctr_2.convert_from_pinhole_camera_parameters(parameters_2)

    print('pause')
    while True:
        for pcd_1_item in points_pcd_1_list:
            vis_1.update_geometry(pcd_1_item)
        if not vis_1.poll_events():
            break
        vis_1.update_renderer()
        for pcd_2_item in points_pcd_2_list:
            vis_2.update_geometry(pcd_2_item)
        if not vis_2.poll_events():
            break
        vis_2.update_renderer()
    

    vis_1.destroy_window()
    vis_2.destroy_window()


def fix_plotly_grid_lines(fig, y_axis = True, x_axis = True):
    if x_axis:
        fig.update_xaxes(minor=dict(ticklen=6, tickcolor="black", showgrid=True))
        fig.update_xaxes(gridcolor='black', zeroline = True, zerolinecolor='black', rangemode = 'tozero')
    if y_axis:
        fig.update_yaxes(gridcolor='black', minor_griddash="dot")
        fig.update_yaxes(zeroline = True, zerolinecolor='black')
    return fig





def dose_point_cloud_with_dose_labels_for_animation_plotly(NN_pts_on_dose_lattice_arr, NN_doses_on_dose_lattice_arr, queried_bx_pts_arr, queried_bx_pts_assigned_doses_arr, num_dose_NN_per_bx_pt, aspect_mode_input = 'data', draw_lines = True, axes_visible = False):
    """
    aspect_mode_input can be one of ( "auto" | "cube" | "data" | "manual" ), follows the plotly fig.update_scenes(aspectmode=<VALUE>) module
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=NN_pts_on_dose_lattice_arr[:,0],
        y=NN_pts_on_dose_lattice_arr[:,1],
        z=NN_pts_on_dose_lattice_arr[:,2],
        mode = 'markers',
        marker = dict(color = 'black', size = 2)
        )
    )
    fig.add_trace(go.Scatter3d(
        x=queried_bx_pts_arr[:,0],
        y=queried_bx_pts_arr[:,1],
        z=queried_bx_pts_arr[:,2],
        mode = 'markers',
        marker = dict(color = 'green', size = 2)
        )
    )

    num_bx_points = queried_bx_pts_arr.shape[0]
    num_NN_dose_points = NN_pts_on_dose_lattice_arr.shape[0]

    if draw_lines == True:
        for i in range(num_bx_points):
            for j in range(num_dose_NN_per_bx_pt):
                bx_pt_and_jth_NN_arr = np.empty((2,3))
                bx_pt_and_jth_NN_arr[0] = queried_bx_pts_arr[i,:]
                bx_pt_and_jth_NN_arr[1] = NN_pts_on_dose_lattice_arr[i*num_dose_NN_per_bx_pt+j,:]
                fig.add_trace(go.Scatter3d(
                    x=bx_pt_and_jth_NN_arr[:,0],
                    y=bx_pt_and_jth_NN_arr[:,1],
                    z=bx_pt_and_jth_NN_arr[:,2],
                    mode = 'lines',
                    line = dict(color = 'black', dash='dot')
                    )
                )
            

    annotations_list = []
    for i in range(num_bx_points):
        annotation_dict = dict(
            showarrow=False,
            x = queried_bx_pts_arr[i,0],
            y = queried_bx_pts_arr[i,1],
            z = queried_bx_pts_arr[i,2],
            text = '{dose_val}'.format(dose_val = round(queried_bx_pts_assigned_doses_arr[i],1)),
            font=dict(color="green", size=12)
            )
        annotations_list.append(annotation_dict.copy())


    for i in range(num_NN_dose_points):
        annotation_dict = dict(
            showarrow=False,
            x = NN_pts_on_dose_lattice_arr[i,0],
            y = NN_pts_on_dose_lattice_arr[i,1],
            z = NN_pts_on_dose_lattice_arr[i,2],
            text = '{dose_val}'.format(dose_val = round(NN_doses_on_dose_lattice_arr[i],1)),
            font=dict(color="black", size=12)
            )
        annotations_list.append(annotation_dict.copy())


    fig.update_layout(
        scene=dict(annotations = annotations_list, aspectmode = aspect_mode_input)
    )
    if axes_visible == False:
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)

    fig.show()


def plotly_3dscatter_arbitrary_number_of_arrays(arrays_to_plot_list, colors_for_arrays_list = [], aspect_mode_input = 'data', title_text = ''):
    """
    aspect_mode_input can be one of ( "auto" | "cube" | "data" | "manual" ), follows the plotly fig.update_scenes(aspectmode=<VALUE>) module
    """
    fig = go.Figure()
    for array_index, pts_array in enumerate(arrays_to_plot_list):
        if len(colors_for_arrays_list) != len(arrays_to_plot_list):
            color_elem = 'rgb'+str(tuple(np.random.randint(low=0,high=255,size=3)))
        else:
            color_elem = colors_for_arrays_list[array_index]
        fig.add_trace(go.Scatter3d(
            x=pts_array[:,0],
            y=pts_array[:,1],
            z=pts_array[:,2],
            mode = 'markers',
            marker = dict(color = color_elem, size = 2)
            )
        )
    
    fig.update_layout(
        scene=dict(aspectmode = aspect_mode_input),
        title = dict(text = str(title_text))
        )    
    
    fig.show()



def create_colored_lines_point_cloud(directions, centroids, lengths, colors):
    """
    Creates a line set in Open3D where each line can have its own color.
    
    Parameters:
    - directions (list of np.array): Direction vectors of the lines.
    - centroids (list of np.array): Centroid positions of each line.
    - lengths (list of float): Lengths of each line.
    - colors (list of list): Colors for each line in RGB format.
    
    Returns:
    - o3d.geometry.LineSet: An Open3D line set object with colored lines.
    """
    points = []
    lines = []
    line_colors = []

    point_index = 0
    for direction, centroid, length, color in zip(directions, centroids, lengths, colors):
        # Normalize the direction vector
        direction = np.array(direction)
        norm_direction = direction / np.linalg.norm(direction)
        
        # Calculate endpoints
        half_length = length / 2
        point1 = centroid + norm_direction * half_length
        point2 = centroid - norm_direction * half_length
        
        # Add endpoints to the points list
        points.append(point1)
        points.append(point2)
        
        # Add line indices (these are indices in the points list)
        lines.append([point_index, point_index + 1])
        point_index += 2
        
        # Add the color for this line
        line_colors.append(color)

    # Create an Open3D line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))
    
    return line_set


def create_colored_origin_axes_o3d_lineset(colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                    lengths = [100,100,100]):
    # add the axes directions to the pointclouds
    directions = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0,0,1])]
    centroids = [np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])]
    
    axes_line_set = create_colored_lines_point_cloud(directions, centroids, lengths, colors)

    return axes_line_set



def dataframe_to_point_cloud_colored_by_value(df, 
                                              x_col_name, 
                                              y_col_name, 
                                              z_col_name, 
                                              color_value_column_name, 
                                              colormap='RdYlGn', 
                                              filter_below_threshold=False, 
                                              threshold=0.05):
    # Extract coordinates
    points = df[[x_col_name, y_col_name, z_col_name]].values
    
    # Reference column for coloring and possibly filtering
    proportions = df[color_value_column_name].values

    if filter_below_threshold:
        # Filter points where proportions are below the threshold
        mask = proportions >= threshold
        points = points[mask]
        proportions = proportions[mask]

    # Normalize 'Proportion of normal dist points contained' for coloring
    min_val = np.min(proportions)
    max_val = np.max(proportions)
    colors = (proportions - min_val) / (max_val - min_val)  # Normalize between 0 and 1
    
    # Map normalized values to a colormap (e.g., viridis)
    cm = plt.get_cmap(colormap)
    colors = cm(colors)[:, :3]  # Ignore alpha channel from matplotlib colors

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd



def dataframe_to_point_cloud_simple(df, 
                                    x_col_name, 
                                    y_col_name, 
                                    z_col_name,
                                    pcd_color = np.array([0,0,0])):
    # Extract coordinates
    points = df[[x_col_name, y_col_name, z_col_name]].values
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Paint the entire point cloud a single color
    pcd.paint_uniform_color(pcd_color)  # pcd_color should be a list or array of 3 elements [R, G, B]

    return pcd













### MR 

def create_MR_point_cloud(data_all_slices_2d_arr, color_flattening_degree, paint_mr_color = True):
    point_cloud = o3d.geometry.PointCloud()
    num_points = data_all_slices_2d_arr.shape[0]
    position_data_all_slices_2d_arr = data_all_slices_2d_arr[:,0:3]
    mr_val_data_all_slices_2d_arr = data_all_slices_2d_arr[:,3]
    max_MR_val = np.amax(mr_val_data_all_slices_2d_arr)
    min_MR_val = np.amin(mr_val_data_all_slices_2d_arr)
    point_cloud.points = o3d.utility.Vector3dVector(position_data_all_slices_2d_arr)
    if paint_mr_color == True:
        root = 1/color_flattening_degree
        pcd_color_arr = np.zeros((num_points,3))
        mr_val_data_all_slices_2d_arr_nth_rooted = np.power(mr_val_data_all_slices_2d_arr,root)
        max_MR_val_nth_rooted = np.power(max_MR_val,root)
        pcd_color_arr[:,0] = mr_val_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr[:,2] = mr_val_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr = pcd_color_arr/max_MR_val_nth_rooted
        pcd_color_arr[:,2] = 1 - pcd_color_arr[:,2]
        #pcd_color_arr[pcd_color_arr<0.1] = 1. # set all points less than a threshold to be white
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color_arr)
    else:
        pcd_color = np.array([0,0,0]) # paint everything black
        point_cloud.paint_uniform_color(pcd_color)

    return point_cloud




def create_thresholded_MR_ADC_point_cloud(data_all_slices_2d_arr, color_flattening_degree, paint_mr_color = True, lower_bound = None, upper_bound = None, z_val_range_list = None):
    point_cloud = o3d.geometry.PointCloud()
    mr_val_data_all_slices_2d_arr = data_all_slices_2d_arr[:,3]
    min_og_mr_val = np.amin(mr_val_data_all_slices_2d_arr)
    max_og_mr_val = np.amax(mr_val_data_all_slices_2d_arr)
    if lower_bound == None:
        lower_bound = min_og_mr_val
    if upper_bound == None:
        upper_bound = max_og_mr_val

    data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr[data_all_slices_2d_arr[:, 3]  > float(lower_bound)]
    data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr_thresholded[data_all_slices_2d_arr_thresholded[:, 3]  < float(upper_bound)]
    
    if z_val_range_list == None:
        pass
    else:
        lower_z_val = z_val_range_list[0]
        upper_z_val = z_val_range_list[1]
        data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr_thresholded[data_all_slices_2d_arr_thresholded[:, 2]  <= float(upper_z_val)]
        data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr_thresholded[data_all_slices_2d_arr_thresholded[:, 2]  >= float(lower_z_val)]




    num_points = data_all_slices_2d_arr_thresholded.shape[0]
    position_data_all_slices_2d_arr = data_all_slices_2d_arr_thresholded[:,0:3]
    mr_val_data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr_thresholded[:,3]

    point_cloud.points = o3d.utility.Vector3dVector(position_data_all_slices_2d_arr)
    if paint_mr_color == True:
        root = 1/color_flattening_degree
        pcd_color_arr = np.zeros((num_points,3))
        mr_val_data_all_slices_2d_arr_thresholded_nth_rooted = np.power(mr_val_data_all_slices_2d_arr_thresholded,root)

        renormalized_data = misc_tools.min_max_normalize_data(mr_val_data_all_slices_2d_arr_thresholded_nth_rooted)

        pcd_color_arr[:,0] = renormalized_data
        pcd_color_arr[:,1] = 0
        pcd_color_arr[:,2] = 1

        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color_arr)
    else:
        pcd_color = np.array([0,0,0]) # paint everything black
        point_cloud.paint_uniform_color(pcd_color)

    return point_cloud

def create_thresholded_MR_point_cloud(data_all_slices_2d_arr, color_flattening_degree, paint_mr_color = True, lower_bound_percent = 10):
    point_cloud = o3d.geometry.PointCloud()
    mr_val_data_all_slices_2d_arr = data_all_slices_2d_arr[:,3]
    max_mr_val_og = np.amax(mr_val_data_all_slices_2d_arr)
    data_all_slices_2d_arr_thresholded = data_all_slices_2d_arr[data_all_slices_2d_arr[:, 3]  > float(max_mr_val_og*lower_bound_percent/100)]
    num_points = data_all_slices_2d_arr_thresholded.shape[0]
    position_data_all_slices_2d_arr = data_all_slices_2d_arr_thresholded[:,0:3]
    mr_val_data_all_slices_2d_arr = data_all_slices_2d_arr_thresholded[:,3]
    min_mr_val = np.amin(mr_val_data_all_slices_2d_arr)
    point_cloud.points = o3d.utility.Vector3dVector(position_data_all_slices_2d_arr)
    if paint_mr_color == True:
        root = 1/color_flattening_degree
        pcd_color_arr = np.zeros((num_points,3))
        mr_val_data_all_slices_2d_arr_nth_rooted = np.power(mr_val_data_all_slices_2d_arr,root)
        max_mr_val_og_nth_rooted = np.power(max_mr_val_og,root)
        pcd_color_arr[:,0] = mr_val_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr[:,2] = mr_val_data_all_slices_2d_arr_nth_rooted
        pcd_color_arr = pcd_color_arr/max_mr_val_og_nth_rooted
        pcd_color_arr[:,2] = 1 - pcd_color_arr[:,2]
        #pcd_color_arr[pcd_color_arr<0.1] = 1. # set all points less than a threshold to be white
        point_cloud.colors = o3d.utility.Vector3dVector(pcd_color_arr)
    else:
        pcd_color = np.array([0,0,0]) # paint everything black
        point_cloud.paint_uniform_color(pcd_color)

    return point_cloud


