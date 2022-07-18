from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import gc


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

def arb_threeD_scatter_plotter_list(data_and_color,**text):
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