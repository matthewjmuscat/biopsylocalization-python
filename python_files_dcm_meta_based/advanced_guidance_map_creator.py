from scipy.spatial import cKDTree
import pandas
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance_matrix
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import cupy as cp
import open3d as o3d
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import point_containment_tools
import plotting_funcs
import copy
import misc_tools
import math
import warnings

def normal_vector(p1, p2, p3):
    v1 = p2 - p1
    v2 = p3 - p1
    return np.cross(v1, v2)

def calculate_rotation_matrix(from_vec, to_vec):
    v = np.cross(from_vec, to_vec)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, to_vec)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        return np.eye(3)  # The vectors are parallel
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / s**2)



def calculate_rotation_matrix_and_euler_ZYX_tait_bryan_order(from_vec, to_vec):
    v = np.cross(from_vec, to_vec)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, to_vec)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        rotation_matrix = np.eye(3)  # The vectors are parallel
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / s**2)
    
    # Calculate Euler angles from rotation matrix
    sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
    else:
        x = np.arctan2(-rotation_matrix[1,2], rotation_matrix[1,1])
        y = np.arctan2(-rotation_matrix[2,0], sy)
        z = 0

    euler_angles = np.array([x, y, z]) * (180/np.pi)  # Convert from radians to degrees
    return rotation_matrix, euler_angles


def calculate_rotation_matrix_and_euler_XYZ_tait_bryan_order(from_vec, to_vec):
    v = np.cross(from_vec, to_vec)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, to_vec)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        rotation_matrix = np.eye(3)  # The vectors are parallel
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / s**2)

    # Calculate Euler angles from rotation matrix assuming Tait-Bryan X-Y-Z order
    if rotation_matrix[2,1] < 1:
        if rotation_matrix[2,1] > -1:
            y = np.arcsin(rotation_matrix[2,0])
            x = np.arctan2(-rotation_matrix[2,1], rotation_matrix[2,2])
            z = np.arctan2(-rotation_matrix[1,0], rotation_matrix[0,0])
        else:
            # Not a unique solution: z - x = atan2(-r12, r11)
            y = -np.pi / 2
            x = -np.arctan2(rotation_matrix[0,1], rotation_matrix[0,0])
            z = 0
    else:
        # Not a unique solution: z + x = atan2(-r12, r11)
        y = np.pi / 2
        x = np.arctan2(rotation_matrix[0,1], rotation_matrix[0,0])
        z = 0

    euler_angles = np.array([x, y, z]) * (180 / np.pi)  # Convert from radians to degrees
    return rotation_matrix, euler_angles


def calculate_rotation_matrix_and_euler_ZYX_order_tait_bryan_extrinsic(from_vec, to_vec):
    v = np.cross(from_vec, to_vec)
    s = np.linalg.norm(v)
    c = np.dot(from_vec, to_vec)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        rotation_matrix = np.eye(3)  # The vectors are parallel
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / s**2)
    
    # Calculate Euler angles from rotation matrix
    # Handle gimbal lock
    if abs(rotation_matrix[2, 0]) != 1:
        y = -np.arcsin(rotation_matrix[2, 0])  # Pitch
        x = np.arctan2(rotation_matrix[2, 1] / np.cos(y), rotation_matrix[2, 2] / np.cos(y))  # Roll
        z = np.arctan2(rotation_matrix[1, 0] / np.cos(y), rotation_matrix[0, 0] / np.cos(y))  # Yaw
    else:
        # Gimbal lock occurs
        y = np.pi / 2 if rotation_matrix[2, 0] == -1 else -np.pi / 2  # Pitch
        z = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])  # Yaw
        x = 0  # Roll is undefined, assume 0

    euler_angles = np.degrees([x, y, z])  # Convert from radians to degrees and return in order of Z, Y, X
    convention = 'Tait-Bryan Z-Y-X (Extrinsic)'
    return rotation_matrix, euler_angles, convention

    

def generate_grid_dataframe(point1, point2, grid_spacing, entire_lattice_df, sp_dil_optimal_coordinate):
    p1 = np.array(sp_dil_optimal_coordinate)
    p2 = np.array(point1)
    p3 = np.array(point2)

    normal = -normal_vector(p1, p2, p3)
    normal /= np.linalg.norm(normal)  # Normalize the normal vector
    
    # Define the rotation from the X-axis to the normal
    #rotation_matrix = calculate_rotation_matrix(np.array([1, 0, 0]), normal)
    rotation_matrix, euler_angles, euler_convention_str = calculate_rotation_matrix_and_euler_ZYX_order_tait_bryan_extrinsic(np.array([1, 0, 0]), normal)

    # Define grid size and spacing based on the desired coverage area
    size = 2 * max(np.linalg.norm(p2-p1), np.linalg.norm(p3-p1))
    num_points = int(size / grid_spacing)
    x_coords = np.zeros((num_points, num_points))
    y_coords, z_coords = np.meshgrid(np.linspace(-size / 2, size / 2, num_points),
                                     np.linspace(-size / 2, size / 2, num_points))

    grid_points = np.column_stack((x_coords.ravel(), y_coords.ravel(), z_coords.ravel()))

    # Rotate grid points to align with the normal
    grid_points = np.dot(grid_points, rotation_matrix.T) + p1

    # KD-tree for nearest neighbor search for these points within entire_lattice_df
    #tree = cKDTree(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
    #                                  'Test location (Prostate centroid origin) (Y)',
    #                                  'Test location (Prostate centroid origin) (Z)']].values, leafsize = 400)
    #distances, indices = tree.query(grid_points, k=1)  # Ensure single closest point
    #values = entire_lattice_df['Proportion of normal dist points contained'].take(indices).to_numpy()

    neighbors = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size = 1,n_jobs = -1).fit(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
                                     'Test location (Prostate centroid origin) (Y)',
                                     'Test location (Prostate centroid origin) (Z)']].values)
    distances, indices = neighbors.kneighbors(grid_points)
    values = entire_lattice_df['Proportion of normal dist points contained'].take(indices.flatten()).to_numpy()

    # indices, min_distances = brute_force_scipy(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
    #                                   'Test location (Prostate centroid origin) (Y)',
    #                                   'Test location (Prostate centroid origin) (Z)']].values, grid_points)

    # values = entire_lattice_df['Proportion of normal dist points contained'].take(indices).to_numpy()


    #indices, min_distances = brute_force_gpu(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
                                    #'Test location (Prostate centroid origin) (Y)',
                                    #'Test location (Prostate centroid origin) (Z)']].values, grid_points)
    # indices, min_distances = brute_force_gpu_chunked(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
    #                                 'Test location (Prostate centroid origin) (Y)',
    #                                 'Test location (Prostate centroid origin) (Z)']].values, 
    #                                 grid_points, 
    #                                 max_chunk_size=10000)
    # values = entire_lattice_df['Proportion of normal dist points contained'].take(indices).to_numpy()

    # Create DataFrame for the grid points and associated values
    grid_df = pandas.DataFrame(grid_points, columns=['Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)'])
    grid_df['Proportion of normal dist points contained'] = values

    return grid_df, normal, euler_angles, euler_convention_str


def brute_force_scipy(data_points, query_points):
    dist_matrix = distance_matrix(query_points, data_points)
    indices = np.argmin(dist_matrix, axis=1)
    min_distances = np.min(dist_matrix, axis=1)
    return indices, min_distances



def brute_force_gpu(data_points, query_points):
    """
    Perform brute force nearest neighbor search using GPU acceleration with CuPy.

    Parameters:
        data_points (numpy.ndarray): Array of data points, shape (N, 3).
        query_points (numpy.ndarray): Array of query points to find nearest neighbors for, shape (M, 3).

    Returns:
        indices (numpy.ndarray): Indices of the nearest neighbors in data_points for each query point.
        min_distances (numpy.ndarray): Minimum distances to the nearest neighbors.
    """
    # Convert numpy arrays to CuPy arrays for GPU computation
    data_points_gpu = cp.asarray(data_points)
    query_points_gpu = cp.asarray(query_points)

    # Calculate distance matrix: ||qi - dj||^2 = ||qi||^2 + ||dj||^2 - 2 * qi * dj
    # Broadcasting sum of squares of data points and query points
    data_squares = cp.sum(data_points_gpu**2, axis=1)
    query_squares = cp.sum(query_points_gpu**2, axis=1)
    distance_matrix = cp.sqrt(query_squares[:, None] + data_squares - 2 * cp.dot(query_points_gpu, data_points_gpu.T))

    # Find the index of the minimum distance for each query point
    indices = cp.argmin(distance_matrix, axis=1)
    min_distances = cp.min(distance_matrix, axis=1)

    # Convert results back to numpy arrays if necessary for further CPU processing
    return cp.asnumpy(indices), cp.asnumpy(min_distances)


def brute_force_gpu_chunked(data_points, query_points, max_chunk_size=10000):
    """
    Perform nearest neighbor search using a brute-force method with GPU acceleration in chunks.

    Parameters:
        data_points (np.array): An array of shape (N, D) representing N points in D dimensions.
        query_points (np.array): An array of shape (M, D) representing M query points in D dimensions.
        max_chunk_size (int): Maximum size of the distance matrix chunk to fit in GPU memory.

    Returns:
        tuple: Two arrays, indices of the closest data points to each query point, and the corresponding minimum distances.
    """
    max_chunk_size = int(max_chunk_size)
    # Transfer data to GPU
    data_gpu = cp.asarray(data_points)
    query_gpu = cp.asarray(query_points)

    num_queries = query_points.shape[0]
    num_data = data_points.shape[0]

    # Initialize arrays to store the minimum distances and their corresponding indices
    min_distances = cp.full(num_queries, cp.inf, dtype=cp.float32)
    min_indices = cp.full(num_queries, -1, dtype=cp.int32)

    # Process chunks of the data points
    for start_idx in range(0, num_data, max_chunk_size):
        end_idx = min(start_idx + max_chunk_size, num_data)
        data_chunk = data_gpu[start_idx:end_idx]

        # Compute the distance matrix for the current chunk
        dist_matrix = cp.linalg.norm(query_gpu[:, None, :] - data_chunk[None, :, :], axis=2)

        # Find the minimum distances and indices for the current chunk
        chunk_min_distances = cp.min(dist_matrix, axis=1)
        chunk_min_indices = cp.argmin(dist_matrix, axis=1) + start_idx  # Adjust indices based on the global array

        # Update the global minimum distances and indices
        update_mask = chunk_min_distances < min_distances
        min_distances[update_mask] = chunk_min_distances[update_mask]
        min_indices[update_mask] = chunk_min_indices[update_mask]

    # Transfer results back to host memory
    min_distances = cp.asnumpy(min_distances)
    min_indices = cp.asnumpy(min_indices)

    return min_indices, min_distances


def transform_grid_and_point_for_plotting(grid_df, normal, sp_dil_optimal_coordinate):
    # Calculate rotation matrix to align normal with (1, 0, 0)
    rotation_matrix = calculate_rotation_matrix(normal, np.array([1, 0, 0]))

    # Determine the centroid of the grid
    centroid = grid_df[['Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)']].mean().values

    # Translate grid to the origin, apply rotation, then translate back
    grid_points = grid_df[['Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)']].values
    grid_points_centered = grid_points - centroid
    rotated_points = np.dot(grid_points_centered, rotation_matrix.T) + centroid

    # Update DataFrame
    grid_df['Transformed X'] = rotated_points[:, 0]
    grid_df['Transformed Y'] = rotated_points[:, 1]
    grid_df['Transformed Z'] = rotated_points[:, 2]

    # Apply the same transformation to sp_dil_optimal_coordinate
    optimal_point_centered = sp_dil_optimal_coordinate - centroid
    transformed_optimal_point = np.dot(optimal_point_centered, rotation_matrix.T) + centroid

    return grid_df, transformed_optimal_point, rotation_matrix


def plot_transformed_contour(df, colorbar_title="Containment proportion", colorbar_title_font_size=12):
    # Create contour plot
    fig = go.Figure()
    fig.add_trace(go.Contour(
                    z=df['Proportion of normal dist points contained'],
                    x=df['Transformed Z'],
                    y=df['Transformed Y'],
                    colorscale=[[0, 'rgba(255,255,255,0)'], # white for zero
                                [0.25, 'rgba(0,0,255,1)'], # blue starts just above zero
                                [0.9, 'rgba(255,0,0,1)'], # red near the top
                                [1, 'rgba(0,255,0,1)']], # green at the maximum
                    zmax = 1,
                    zmin = 0,
                    autocontour = False,
                    contours = go.contour.Contours(type = 'levels', showlines = True, coloring = 'heatmap', showlabels = False, size = 0.1),
                    connectgaps = False, 
                    colorbar = go.contour.ColorBar(len = 0.5, title=dict(text=colorbar_title, font=dict(size=colorbar_title_font_size)))
                ))
    fig.update_layout(title='TRUS plane',
                      xaxis_title='Craniocaudal axis Z\' (Sup/Inf) [mm]', yaxis_title='Frontal/Anteroposterior axis Y\' (Post/Ant) [mm]')
    return fig

def plot_transverse_contour(fig_input, df, colorbar_title="Containment proportion", colorbar_title_font_size=12):
    # Create contour plot
    fig = fig_input
    fig.add_trace(go.Contour(
                    z=df['Proportion of normal dist points contained'],
                    x=df['Test location (Prostate centroid origin) (X)'],
                    y=df['Test location (Prostate centroid origin) (Y)'],
                    colorscale=[[0, 'rgba(255,255,255,0)'], # white for zero
                                [0.25, 'rgba(0,0,255,1)'], # blue starts just above zero
                                [0.9, 'rgba(255,0,0,1)'], # red near the top
                                [1, 'rgba(0,255,0,1)']], # green at the maximum
                    zmax = 1,
                    zmin = 0,
                    autocontour = False,
                    contours = go.contour.Contours(type = 'levels', showlines = True, coloring = 'heatmap', showlabels = False, size = 0.1),
                    connectgaps = False, 
                    colorbar = go.contour.ColorBar(len = 0.5, title=dict(text=colorbar_title, font=dict(size=colorbar_title_font_size)))
                ))
    fig.update_layout(title='Transverse (Max) plane',
                      xaxis_title='Frontal axis X (Right/Left) [mm]', yaxis_title='Anteroposterior axis Y (Post/Ant) [mm]')
    return fig

def add_points_to_plot(fig, points_arr, legend_name = 'Data', color = "orange", size = 10):
    """Add transformed base point to the plot with an appropriate label."""

    points_arr = np.atleast_2d(points_arr)

    # Add the transformed point to the plot
    fig.add_trace(go.Scatter(
        x=points_arr[:,0],  
        y=points_arr[:,1],  
        mode='markers',
        marker=dict(color=color, size=size),
        name=legend_name  # This sets the name in the legend
    ))

    return fig

def add_points_to_plot_v2(fig, 
                          points_arr, 
                          text="", 
                          color="orange", # this will be overridden if color_index != None!
                          size=10, 
                          symbol = 'circle', 
                          custom_height = 0, 
                          text_color = "#ffffff", 
                          text_box_bg_color = "rgba(50, 50, 50, 0.8)", 
                          text_box_border_color = "rgba(255, 255, 255, 0.2)",
                          font_size = 12,
                          legend_name = "Data",
                          color_index = None,
                          annotation_x_offset = 0.0,
                          annotation_direction = "up"):
    """Add transformed base point to the plot with an appropriate label, including a 'hockey stick line'."""
    colors = ['#333333', '#FF7F50', '#008080']  # Dark charcoal gray, coral, teal

    if color_index == None:
        pass
    else:
        color = colors[color_index % len(colors)]
    
    points_arr = np.atleast_2d(points_arr)

    # Check if the y-axis has a defined range; if not, assume a default range
    y_axis_range = fig.layout.yaxis.range if fig.layout.yaxis.range else [0, 1]
    x_axis_range = fig.layout.xaxis.range if fig.layout.xaxis.range else [0, 1]
    top_y = y_axis_range[1]
    bottom_y = y_axis_range[0]
    left_x = y_axis_range[0]
    right_x = y_axis_range[1]
    annotation_tick_height = (top_y - bottom_y)*0.05
    if annotation_direction == "down":
        vert_line_end = bottom_y + (top_y - bottom_y)*0.2
        stick_end_y = bottom_y + annotation_tick_height + custom_height
        yanchor = 'top'
    else:
        vert_line_end = top_y - (top_y - bottom_y)*0.2
        stick_end_y = top_y - annotation_tick_height + custom_height
        yanchor = 'bottom'
    slope = abs((top_y - bottom_y)/(left_x-right_x))

    # Add points to the plot
    fig.add_trace(go.Scatter(
        x=points_arr[:, 0],
        y=points_arr[:, 1],
        mode='markers',
        marker=dict(color=color, size=size, symbol = symbol),
        name=legend_name  # This sets the name in the legend

    ))

    # Loop through each point to add labels and lines
    for point in points_arr:
        x, y = point[0], point[1]
        x_ann = x + annotation_x_offset
        #label_y = top_y + 0.02 * (top_y - y_axis_range[0])  # Offset text slightly above the top
        # Add annotation at the top
        fig.add_annotation(
            x=(stick_end_y-(vert_line_end-slope*x_ann))/slope, y=stick_end_y,
            text=text,
            showarrow=False,
            xanchor='left',  # Anchoring text to the left of the point
            yanchor=yanchor,
            align='left',
            font=dict(
            family="Courier New, monospace",
            size=font_size,
            color=text_color),
            bgcolor=text_box_bg_color,
            bordercolor=text_box_border_color,
            borderpad=4
        )


        # Add a 'hockey stick line' from the point to the text
        fig.add_shape(type="line",
                      x0=x, y0=y, x1=x, y1=vert_line_end,  # Vertical line segment
                      line=dict(color=color, width=2))
        fig.add_shape(type="line",
                      x0=x, y1=stick_end_y, x1=(stick_end_y-(vert_line_end-slope*x_ann))/slope, y0=vert_line_end,  # Horizontal line segment
                      line=dict(color=color, width=2))

    return fig

def add_compact_fire_positions_table(fig,
                                     fire_rows,
                                     position='auto',
                                     frame_label="Transducer plane frame (Z', Y')",
                                     optimal_row=None):
    """
    Add a compact fire-position summary table as a single annotation.
    If position is "auto", choose the least-crowded corner in paper space.
    Optionally include optimal-point information in the same table.
    """
    if len(fire_rows) == 0:
        return fig

    def _strip_markup(text):
        text_str = str(text)
        for token in ["<b>", "</b>", "<br>", "<br/>", "<br />"]:
            text_str = text_str.replace(token, "\n" if "br" in token else "")
        return text_str

    def _estimate_text_box_size(lines, font_size):
        line_count = max(1, len(lines))
        max_chars = max(len(line) for line in lines) if lines else 1
        width = min(0.52, 0.08 + max_chars * font_size * 0.00075)
        height = min(0.52, 0.03 + line_count * font_size * 0.0038)
        return width, height

    def _bbox_from_anchor(x, y, width, height, xanchor, yanchor):
        if xanchor == "right":
            xmin, xmax = x - width, x
        elif xanchor == "center":
            xmin, xmax = x - width / 2, x + width / 2
        else:
            xmin, xmax = x, x + width

        if yanchor == "top":
            ymin, ymax = y - height, y
        elif yanchor == "middle":
            ymin, ymax = y - height / 2, y + height / 2
        else:
            ymin, ymax = y, y + height
        return np.array([xmin, xmax, ymin, ymax], dtype=float)

    def _intersection_area(box_a, box_b):
        x_overlap = max(0.0, min(box_a[1], box_b[1]) - max(box_a[0], box_b[0]))
        y_overlap = max(0.0, min(box_a[3], box_b[3]) - max(box_a[2], box_b[2]))
        return x_overlap * y_overlap

    def _bbox_edge_distance(box_a, box_b):
        dx = max(0.0, max(box_b[0] - box_a[1], box_a[0] - box_b[1]))
        dy = max(0.0, max(box_b[2] - box_a[3], box_a[2] - box_b[3]))
        return math.sqrt(dx * dx + dy * dy)

    def _axis_name_from_ref(axis_ref, axis_letter):
        suffix = axis_ref[1:]
        return f"{axis_letter}axis{suffix}" if suffix else f"{axis_letter}axis"

    def _axis_domain(axis_ref, axis_letter):
        axis_obj = getattr(fig.layout, _axis_name_from_ref(axis_ref, axis_letter), None)
        if axis_obj is not None and getattr(axis_obj, "domain", None) is not None:
            domain = axis_obj.domain
            return float(domain[0]), float(domain[1])
        primary_axis = getattr(fig.layout, f"{axis_letter}axis", None)
        if primary_axis is not None and getattr(primary_axis, "domain", None) is not None:
            domain = primary_axis.domain
            return float(domain[0]), float(domain[1])
        return 0.0, 1.0

    def _axis_range(axis_ref, axis_letter):
        axis_obj = getattr(fig.layout, _axis_name_from_ref(axis_ref, axis_letter), None)
        if axis_obj is not None and getattr(axis_obj, "range", None) is not None:
            range_vals = axis_obj.range
            if len(range_vals) >= 2:
                return float(range_vals[0]), float(range_vals[1])

        values = []
        trace_axis_key = f"{axis_letter}axis"
        trace_data_key = axis_letter
        for tr in fig.data:
            tr_axis_ref = getattr(tr, trace_axis_key, None) or axis_letter
            if tr_axis_ref != axis_ref:
                continue
            tr_vals = getattr(tr, trace_data_key, None)
            if tr_vals is None:
                continue
            tr_arr = np.asarray(tr_vals).ravel()
            if tr_arr.size == 0:
                continue
            try:
                tr_numeric = tr_arr.astype(float, copy=False)
            except (TypeError, ValueError):
                continue
            numeric_vals = tr_numeric[np.isfinite(tr_numeric)]
            if numeric_vals.size > 0:
                values.append(numeric_vals)
        if values:
            merged = np.concatenate(values)
            return float(np.min(merged)), float(np.max(merged))
        return None

    def _data_to_paper(value, axis_ref, axis_letter):
        axis_range = _axis_range(axis_ref, axis_letter)
        if axis_range is None:
            return None
        r0, r1 = axis_range
        if abs(r1 - r0) < 1e-9:
            rel = 0.5
        else:
            rel = (float(value) - r0) / (r1 - r0)
        d0, d1 = _axis_domain(axis_ref, axis_letter)
        return d0 + rel * (d1 - d0)

    def _annotation_bbox(annotation):
        ann_text = annotation.text or ""
        ann_lines = _strip_markup(ann_text).split("\n")
        ann_font_size = getattr(annotation.font, "size", None) or 12
        ann_w, ann_h = _estimate_text_box_size(ann_lines, ann_font_size)

        ann_xref = getattr(annotation, "xref", None) or "x"
        ann_yref = getattr(annotation, "yref", None) or "y"
        ann_x = getattr(annotation, "x", None)
        ann_y = getattr(annotation, "y", None)
        if ann_x is None or ann_y is None:
            return None

        if ann_xref == "paper":
            x_paper = float(ann_x)
        elif isinstance(ann_xref, str) and ann_xref.startswith("x"):
            x_paper = _data_to_paper(float(ann_x), ann_xref, "x")
        else:
            return None
        if ann_yref == "paper":
            y_paper = float(ann_y)
        elif isinstance(ann_yref, str) and ann_yref.startswith("y"):
            y_paper = _data_to_paper(float(ann_y), ann_yref, "y")
        else:
            return None
        if x_paper is None or y_paper is None:
            return None

        xanchor = getattr(annotation, "xanchor", None) or "center"
        yanchor = getattr(annotation, "yanchor", None) or "middle"
        if xanchor == "auto":
            xanchor = "center"
        if yanchor == "auto":
            yanchor = "middle"
        return _bbox_from_anchor(x_paper, y_paper, ann_w, ann_h, xanchor, yanchor)

    def _colorbar_bbox(colorbar):
        orientation = getattr(colorbar, "orientation", None) or "v"
        x = float(getattr(colorbar, "x", None) if getattr(colorbar, "x", None) is not None else 1.02)
        y = float(getattr(colorbar, "y", None) if getattr(colorbar, "y", None) is not None else 0.5)
        bar_len = float(getattr(colorbar, "len", None) if getattr(colorbar, "len", None) is not None else 1.0)

        thickness_mode = getattr(colorbar, "thicknessmode", None) or "pixels"
        thickness_val = getattr(colorbar, "thickness", None)
        if thickness_mode == "fraction" and thickness_val is not None:
            thickness = float(thickness_val)
        else:
            thickness = 0.06

        if orientation == "h":
            width = max(0.12, min(1.0, bar_len))
            height = max(0.03, min(0.14, thickness))
            xanchor = getattr(colorbar, "xanchor", None) or "center"
            yanchor = getattr(colorbar, "yanchor", None) or "middle"
        else:
            width = max(0.03, min(0.14, thickness))
            height = max(0.12, min(1.0, bar_len))
            xanchor = getattr(colorbar, "xanchor", None) or "left"
            yanchor = getattr(colorbar, "yanchor", None) or "middle"
        return _bbox_from_anchor(x, y, width, height, xanchor, yanchor)

    def _legend_bbox():
        if fig.layout.showlegend is False:
            return None
        legend = getattr(fig.layout, "legend", None)
        if legend is None:
            return None

        x = float(getattr(legend, "x", None) if getattr(legend, "x", None) is not None else 1.02)
        y = float(getattr(legend, "y", None) if getattr(legend, "y", None) is not None else 1.0)
        xanchor = getattr(legend, "xanchor", None) or ("left" if x >= 0.5 else "right")
        yanchor = getattr(legend, "yanchor", None) or ("top" if y >= 0.5 else "bottom")
        if yanchor == "auto":
            yanchor = "top" if y >= 0.5 else "bottom"

        legend_labels = []
        for tr in fig.data:
            if getattr(tr, "showlegend", None) is False:
                continue
            trace_name = getattr(tr, "name", None)
            if trace_name:
                legend_labels.append(str(trace_name))
        if not legend_labels:
            return None

        n_labels = len(legend_labels)
        max_chars = max(len(label) for label in legend_labels)
        legend_width = min(0.45, 0.10 + max_chars * 0.0065)
        legend_height = min(0.75, 0.06 + n_labels * 0.035)
        return _bbox_from_anchor(x, y, legend_width, legend_height, xanchor, yanchor)

    def _collect_obstacle_bboxes():
        obstacles = []
        if fig.layout.annotations:
            for ann in fig.layout.annotations:
                ann_bbox = _annotation_bbox(ann)
                if ann_bbox is not None:
                    obstacles.append(ann_bbox)

        legend_box = _legend_bbox()
        if legend_box is not None:
            obstacles.append(legend_box)

        for tr in fig.data:
            if hasattr(tr, "colorbar") and tr.colorbar:
                obstacles.append(_colorbar_bbox(tr.colorbar))
        return obstacles

    def _collect_trace_points(max_points_per_trace=250):
        points = []
        for tr in fig.data:
            x_vals = getattr(tr, "x", None)
            y_vals = getattr(tr, "y", None)
            if x_vals is None or y_vals is None:
                continue
            x_arr = np.asarray(x_vals).ravel()
            y_arr = np.asarray(y_vals).ravel()
            n = min(x_arr.size, y_arr.size)
            if n == 0:
                continue

            x_ref = getattr(tr, "xaxis", None) or "x"
            y_ref = getattr(tr, "yaxis", None) or "y"
            stride = max(1, n // max_points_per_trace)
            for i in range(0, n, stride):
                x_val = x_arr[i]
                y_val = y_arr[i]
                try:
                    x_paper = _data_to_paper(float(x_val), x_ref, "x")
                    y_paper = _data_to_paper(float(y_val), y_ref, "y")
                except (TypeError, ValueError):
                    continue
                if x_paper is None or y_paper is None:
                    continue
                if np.isfinite(x_paper) and np.isfinite(y_paper):
                    points.append((float(x_paper), float(y_paper)))
        return points

    def _score_candidate(candidate_bbox, obstacles, points):
        score = 0.0
        xmin, xmax, ymin, ymax = candidate_bbox

        out_of_bounds = (
            max(0.0, -xmin) + max(0.0, xmax - 1.0) +
            max(0.0, -ymin) + max(0.0, ymax - 1.0)
        )
        score += out_of_bounds * 1000.0

        for obstacle in obstacles:
            overlap = _intersection_area(candidate_bbox, obstacle)
            if overlap > 0:
                score += 1000.0 + overlap * 25000.0
            else:
                gap = _bbox_edge_distance(candidate_bbox, obstacle)
                if gap < 0.04:
                    score += (0.04 - gap) * 220.0

        expanded = np.array([xmin - 0.01, xmax + 0.01, ymin - 0.01, ymax + 0.01], dtype=float)
        for px, py in points:
            if expanded[0] <= px <= expanded[1] and expanded[2] <= py <= expanded[3]:
                score += 4.0

        return score

    position_settings = {
        "top right": {"x": 0.99, "y": 0.99, "xanchor": "right", "yanchor": "top"},
        "top left": {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
        "bottom right": {"x": 0.99, "y": 0.01, "xanchor": "right", "yanchor": "bottom"},
        "bottom left": {"x": 0.01, "y": 0.01, "xanchor": "left", "yanchor": "bottom"},
        "middle right": {"x": 0.99, "y": 0.5, "xanchor": "right", "yanchor": "middle"},
        "middle left": {"x": 0.01, "y": 0.5, "xanchor": "left", "yanchor": "middle"},
        "middle top": {"x": 0.5, "y": 0.99, "xanchor": "center", "yanchor": "top"},
        "middle bottom": {"x": 0.5, "y": 0.01, "xanchor": "center", "yanchor": "bottom"},
    }
    position_normalized = (position or "auto").lower()

    def _fmt_or_dash(value, decimals=2):
        try:
            val = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(val):
            return "-"
        return f"{val:.{decimals}f}"

    def _text_or_dash(value):
        if value is None:
            return "-"
        text_val = str(value).strip()
        return text_val if text_val else "-"

    column_headers = [
        "Type",
        "Optimal hole",
        "Depth (mm)",
        "Z' (mm)",
        "Y' (mm)",
        "Defl (mm)",
        "From apex (mm)",
    ]

    table_rows = []
    for fire_index, fire_row in enumerate(fire_rows):
        table_rows.append({
            "Type": f"Depth setting {fire_index + 1}",
            "Optimal hole": _text_or_dash(fire_row.get("optimal_hole")),
            "Depth (mm)": _fmt_or_dash(fire_row.get("penetration_depth"), decimals=1),
            "Z' (mm)": _fmt_or_dash(fire_row.get("zprime"), decimals=1),
            "Y' (mm)": _fmt_or_dash(fire_row.get("yprime"), decimals=1),
            "Defl (mm)": _fmt_or_dash(fire_row.get("deflection"), decimals=1),
            "From apex (mm)": _fmt_or_dash(fire_row.get("depth_from_apex"), decimals=1),
        })

    if optimal_row is not None:
        table_rows.append({
            "Type": str(optimal_row.get("label") or "Optimal template hole"),
            "Optimal hole": _text_or_dash(optimal_row.get("optimal_hole")),
            "Depth (mm)": "-",
            "Z' (mm)": _fmt_or_dash(optimal_row.get("zprime"), decimals=1),
            "Y' (mm)": _fmt_or_dash(optimal_row.get("yprime"), decimals=1),
            "Defl (mm)": "-",
            "From apex (mm)": "-",
        })

    max_chars_per_col = []
    for col_header in column_headers:
        max_chars = len(col_header)
        for row in table_rows:
            max_chars = max(max_chars, len(str(row[col_header])))
        max_chars_per_col.append(max_chars)

    approx_char_count = sum(max_chars_per_col)
    column_width_weights = []
    for col_header, max_chars in zip(column_headers, max_chars_per_col):
        width_weight = float(max(6, max_chars))
        if col_header == "Type":
            width_weight *= 1.35
        elif col_header == "Optimal hole":
            width_weight *= 1.2
        column_width_weights.append(width_weight)
    plot_x_min, plot_x_max = _axis_domain("x", "x")
    plot_domain_width = max(0.01, float(plot_x_max - plot_x_min))
    table_width = max(0.56, 0.0085 * approx_char_count)
    table_width = min(table_width, plot_domain_width)
    table_height = min(0.84, max(0.18, 0.10 + 0.042 * len(table_rows)))
    if frame_label:
        title_min_width = 0.12 + 0.0068 * len(str(frame_label))
        table_width = max(table_width, title_min_width)
        table_width = min(table_width, plot_domain_width)

    # Reserve title band above table and include it in placement/scoring.
    title_gap = 0.006 if frame_label else 0.0
    title_band_height = 0.045 if frame_label else 0.0
    total_block_height = table_height + title_gap + title_band_height

    if position_normalized == "outside top center":
        # Reserve a top band outside the plotting area and center the table in that band.
        domain_pad = 0.01
        domain_left = float(plot_x_min + domain_pad)
        domain_right = float(plot_x_max - domain_pad)
        if domain_right <= domain_left:
            domain_left = float(plot_x_min)
            domain_right = float(plot_x_max)

        max_width_for_domain = max(0.01, domain_right - domain_left)
        # Use the full available plot-domain width in this mode.
        table_width = max_width_for_domain

        x_center = 0.5 * (plot_x_min + plot_x_max)
        xmin = x_center - table_width / 2.0
        xmax = x_center + table_width / 2.0
        if xmin < domain_left:
            xmax += (domain_left - xmin)
            xmin = domain_left
        if xmax > domain_right:
            xmin -= (xmax - domain_right)
            xmax = domain_right
        xmin = float(np.clip(xmin, domain_left, max(domain_left, domain_right - 0.01)))
        xmax = float(np.clip(xmax, xmin + 0.01, domain_right))

        ymax = 0.99
        ymin = max(0.01, ymax - total_block_height)

        plot_domain_top = float(np.clip(ymin - 0.01, 0.35, 0.95))
        fig.update_layout(
            yaxis=dict(domain=[0, plot_domain_top]),
            yaxis2=dict(domain=[0, plot_domain_top],
                        overlaying="y",
                        side="right",
                        matches="y")
        )
    else:
        if position_normalized == "auto":
            obstacles = _collect_obstacle_bboxes()
            trace_points = _collect_trace_points()
            candidate_names = [
                "top left",
                "top right",
                "bottom left",
                "bottom right",
                "middle left",
                "middle right",
                "middle top",
                "middle bottom",
            ]
            candidate_scores = []
            for candidate_name in candidate_names:
                candidate_settings = position_settings[candidate_name]
                candidate_bbox = _bbox_from_anchor(candidate_settings["x"],
                                                   candidate_settings["y"],
                                                   table_width,
                                                   total_block_height,
                                                   candidate_settings["xanchor"],
                                                   candidate_settings["yanchor"])
                candidate_scores.append((_score_candidate(candidate_bbox, obstacles, trace_points), candidate_name))
            candidate_scores.sort(key=lambda item: item[0])
            pos_settings = position_settings[candidate_scores[0][1]]
        else:
            pos_settings = position_settings.get(position_normalized, position_settings["top left"])

        block_bbox = _bbox_from_anchor(pos_settings["x"],
                                       pos_settings["y"],
                                       table_width,
                                       total_block_height,
                                       pos_settings["xanchor"],
                                       pos_settings["yanchor"])
        xmin, xmax, ymin, ymax = block_bbox
        dx = 0.0
        dy = 0.0
        if xmin < 0.01:
            dx = 0.01 - xmin
        elif xmax > 0.99:
            dx = 0.99 - xmax
        if ymin < 0.01:
            dy = 0.01 - ymin
        elif ymax > 0.99:
            dy = 0.99 - ymax
        xmin += dx
        xmax += dx
        ymin += dy
        ymax += dy

        xmin = float(np.clip(xmin, 0.01, 0.98))
        xmax = float(np.clip(xmax, xmin + 0.01, 0.99))
        ymin = float(np.clip(ymin, 0.01, 0.98))
        ymax = float(np.clip(ymax, ymin + 0.01, 0.99))

    table_ymin = ymin
    table_ymax = ymax - (title_gap + title_band_height)
    table_ymax = float(np.clip(table_ymax, table_ymin + 0.01, 0.99))

    table_cells_by_col = [[row[col_header] for row in table_rows] for col_header in column_headers]
    row_fill_colors = ["rgba(255,255,255,1)" for _ in table_rows]
    if optimal_row is not None and len(row_fill_colors) > 0:
        row_fill_colors[-1] = "rgba(238,238,238,1)"
    fill_colors_by_col = [row_fill_colors[:] for _ in column_headers]

    # Slightly taller rows so larger publication fonts remain legible without clipping.
    header_height = 30
    cell_height = 26

    fig.add_trace(go.Table(
        domain=dict(x=[xmin, xmax], y=[table_ymin, table_ymax]),
        header=dict(
            values=[f"<b>{col_header}</b>" for col_header in column_headers],
            align="center",
            fill_color="rgba(255,255,255,1)",
            line_color="black",
            font=dict(size=11, color="black"),
            height=header_height
        ),
        cells=dict(
            values=table_cells_by_col,
            align=["left", "center", "center", "center", "center", "center", "center"],
            fill_color=fill_colors_by_col,
            line_color="black",
            font=dict(size=10, color="black"),
            height=cell_height
        ),
        columnwidth=column_width_weights
    ))

    if frame_label:
        fig.add_annotation(
            x=(xmin + xmax) / 2.0,
            y=(table_ymax + ymax) / 2.0,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=f"<b>{frame_label}</b>",
            font=dict(size=11, color="black"),
            xanchor="center",
            yanchor="middle",
            bgcolor="rgba(255,255,255,1)",
            bordercolor="rgba(0,0,0,1)",
            borderpad=2
        )

    return fig

def slice_mesh_fast(mesh, plane_normal, plane_point):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    plane_normal = np.asarray(plane_normal)
    plane_point = np.asarray(plane_point)

    # Normalize the plane normal for safety
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Compute the signed distances of all vertices from the plane
    distances = np.dot(vertices - plane_point, plane_normal)

    # List to hold intersection points
    intersection_points = []

    # Process each triangle
    for triangle in triangles:
        pts = vertices[triangle]
        dists = distances[triangle]
        
        # We need to find intersections on the triangle edges
        for i in range(3):
            j = (i + 1) % 3
            if dists[i] * dists[j] < 0:  # If signs are different, there is an intersection
                interp = dists[i] / (dists[i] - dists[j])
                intersection = pts[i] + interp * (pts[j] - pts[i])
                intersection_points.append(intersection)

    # Convert list of intersection points to a NumPy array
    if intersection_points:
        return np.array(intersection_points)
    return np.array([])  # Return an empty array if no intersections


def slice_mesh_fast_v2(mesh, plane_normal, plane_point, epsilon=1e-6):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    plane_normal = np.asarray(plane_normal)
    plane_point = np.asarray(plane_point)

    # Normalize the plane normal to ensure accurate distance calculations
    plane_normal /= np.linalg.norm(plane_normal)

    # Compute the signed distances of all vertices from the plane
    distances = np.dot(vertices - plane_point, plane_normal)

    # List to hold intersection points
    intersection_points = []

    # Process each triangle
    for triangle in triangles:
        pts = vertices[triangle]
        dists = distances[triangle]

        # Check if the entire triangle lies on the plane
        if np.all(np.abs(dists) < epsilon):
            intersection_points.extend(pts)  # Add all vertices of the triangle
            continue

        # Check intersections on the triangle edges
        for i in range(3):
            j = (i + 1) % 3
            if dists[i] * dists[j] < 0:  # If signs are different, there is an intersection
                interp = dists[i] / (dists[i] - dists[j])
                intersection = pts[i] + interp * (pts[j] - pts[i])
                intersection_points.append(intersection)
            elif np.abs(dists[i]) < epsilon and np.abs(dists[j]) < epsilon:
                # Edge is on the plane
                intersection_points.append(pts[i])
                intersection_points.append(pts[j])

    # Convert list of intersection points to a NumPy array
    if intersection_points:
        return np.array(intersection_points)
    return np.array([])  # Return an empty array if no intersections


def transform_points(points, rotation_matrix, grid_df):
    """ Apply rotation and translation to points """
    centroid = grid_df[['Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)']].mean().values
    points_centered = points - centroid
    transformed_points = np.dot(points_centered, rotation_matrix.T) + centroid
    return transformed_points


def plot_transformed_mesh_slice(fig, slice_points, mesh_name, color_input = 'red'):
    if isinstance(color_input, np.ndarray):
        color_str = f"rgb({color_input[0]}, {color_input[1]}, {color_input[2]})" 
        color = color_str
    elif isinstance(color_input, str):
        color = color_input
    # If slice points are provided, plot them
    fig.add_trace(go.Scatter(
            x=slice_points[:, 0],  
            y=slice_points[:, 1],  
            mode='markers',
            marker=dict(color=color, size=3),
            name=mesh_name
        ))

    return fig



"""
def dataframe_to_point_cloud(df, x_col_name, y_col_name, z_col_name, colormap = 'RdYlGn'):
    # Extract coordinates
    points = df[[x_col_name, y_col_name, z_col_name]].values
    
    # Normalize 'Proportion of normal dist points contained' for coloring
    proportions = df['Proportion of normal dist points contained'].values
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
"""

def dataframe_to_point_cloud(df, x_col_name, y_col_name, z_col_name, colormap='RdYlGn', filter_below_threshold=False, threshold=0.05):
    # Extract coordinates
    points = df[[x_col_name, y_col_name, z_col_name]].values
    
    # Reference column for coloring and possibly filtering
    proportions = df['Proportion of normal dist points contained'].values

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

def translate_mesh(mesh, translation_vector):
    """
    Translate an Open3D TriangleMesh object by a given vector.

    Parameters:
    - mesh: open3d.geometry.TriangleMesh, the mesh to be translated
    - translation_vector: array-like, the translation vector (dx, dy, dz)

    Returns:
    - Translated open3d.geometry.TriangleMesh object
    """
    # Convert translation_vector to a numpy array to ensure compatibility
    new_mesh = copy.deepcopy(mesh)
    translation_vector = np.array(translation_vector)
    
    # Translate the vertices
    vertices = np.asarray(new_mesh.vertices)
    translated_vertices = vertices + translation_vector
    
    # Update the mesh vertices with the translated coordinates
    new_mesh.vertices = o3d.utility.Vector3dVector(translated_vertices)
    return new_mesh

def solve_tsp(points):
    """
    Organize points by solving the Traveling Salesman Problem (TSP).

    Parameters:
        points (np.array): Nx2 or Nx3 array of points.

    Returns:
        np.array: Ordered points according to the TSP solution.
    """
    # Calculate the distance matrix
    distance_matrix = squareform(pdist(points, metric='euclidean'))

    # Create a complete graph from the distance matrix
    G = nx.complete_graph(len(points), create_using=nx.Graph())
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            G[i][j]['weight'] = distance_matrix[i][j]

    # Compute a TSP approximate solution using a greedy algorithm
    tsp_path = nx.approximation.greedy_tsp(G, source=0)

    # Order points based on the TSP path
    ordered_points = points[tsp_path]

    return ordered_points



def solve_tsp_google(points):
    """
    Organize points by solving the Traveling Salesman Problem (TSP) using Google OR-Tools.

    Parameters:
        points (np.array): Nx2 or Nx3 array of points.

    Returns:
        np.array: Ordered points according to the TSP solution.
    """
    # Create the data model.
    def create_data_model():
        data = {}
        # Calculate the Euclidean distance between two points
        data['distance_matrix'] = [
            [int(np.linalg.norm(points[i] - points[j]))
             for j in range(len(points))] for i in range(len(points))
        ]
        data['num_vehicles'] = 1
        data['depot'] = 0
        return data

    # Instantiate the data problem.
    data = create_data_model()

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic: Path Cheapest Arc.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)
    if not solution:
        return np.array([])  # No solution found

    # Extract the ordered path from the solution.
    index = routing.Start(0)
    tsp_path = []
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        tsp_path.append(node_index)
        index = solution.Value(routing.NextVar(index))

    # Order points based on the TSP path
    ordered_points = points[tsp_path]

    return ordered_points



def adjust_plot_area(fig, points, margin=5):
    """
    Adjusts the plot area to be within a specified margin around the points.

    Parameters:
        fig (go.Figure): The figure object containing the contour plot.
        points (np.array): Array of points (x, y) representing the prostate contour.
        margin (float): Margin to add around the contour.
    """
    # Determine the bounds of the points
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Set the range for x and y axes
    fig.update_layout(
        xaxis=dict(range=[min_x - margin, max_x + margin]),
        yaxis=dict(range=[min_y - margin, max_y + margin])
    )

    return fig


def adjust_plot_area_and_reverse_axes(fig, points, margin=5, reverse_x = False, reverse_y = False):
    min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
    min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

    # Consider the margin based on reversed axes
    if reverse_x:
        x_range = [max_x + margin, min_x - margin]
    else:
        x_range = [min_x - margin, max_x + margin]
    if reverse_y:
        y_range = [max_y + margin, min_y - margin]
    else:
        y_range = [min_y - margin, max_y + margin]

    # Update the layout with the new ranges
    fig.update_layout(
        xaxis=dict(range=x_range),
        yaxis=dict(range=y_range)
    )
    return fig


def add_x_bounds_with_annotations(fig, 
                                  points, 
                                  y_position = -0.05,
                                  x_offset = -3, 
                                  label_yanchor = "bottom",
                                  max_x_label="Max", 
                                  min_x_label="Min", 
                                  line_color_max='black', 
                                  line_color_min='black', 
                                  line_style_max='solid',
                                  line_style_min='solid', 
                                  line_width=3, 
                                  font_color='black',
                                  label_bg_color="rgba(255, 255, 255, 1)",
                                  label_border_color="rgba(0, 0, 0, 1)",
                                  label_border_pad=3):
    """
    Adds vertical lines and annotations for the minimum and maximum x-values of given points, with annotations placed below the x-axis.

    Parameters:
        fig (go.Figure): The figure object to modify.
        points (np.array): Array of points [n_points, 2] or [n_points, 3] where x is the first column.
        y_position (float): The y-axis position to place annotations (should be below the y-axis range).
        y_range (tuple): Tuple (y_min, y_max) defining the extent of the y-axis if not auto-scaled by plotly.
    """
    # Determine y-limits for line drawing
    if fig.layout.yaxis.range:
        y_range = fig.layout.yaxis.range
    else:
        y_range = [0, 1]
  
    # Calculate minimum and maximum x-values
    min_x = np.min(points[:, 0])
    max_x = np.max(points[:, 0])

    # Draw full-height vertical lines in axis-domain coordinates so they always span
    # bottom-to-top of the visible plot area even if ranges/aspect are changed later.
    fig.add_shape(
        type="line",
        x0=min_x, x1=min_x,
        y0=0, y1=1,
        xref="x", yref="y domain",
        line=dict(color=line_color_min, width=line_width, dash=line_style_min)
    )
    fig.add_shape(
        type="line",
        x0=max_x, x1=max_x,
        y0=0, y1=1,
        xref="x", yref="y domain",
        line=dict(color=line_color_max, width=line_width, dash=line_style_max)
    )

    # Label y-location as a normalized offset from the visual bottom of the axis.
    # 0.0 is at the bottom axis boundary; negative values move below the plot area.
    y_label = y_range[0] + y_position * (y_range[1] - y_range[0])
    fig.add_annotation(
        x=min_x + x_offset, y=y_label,
        text=f"{min_x_label}",
        showarrow=False,
        xanchor="left",
        yanchor=label_yanchor,
        xref="x",
        yref="y",
        textangle=0,
        font=dict(color=font_color, size = 16),
        bgcolor=label_bg_color,
        bordercolor=label_border_color,
        borderpad=label_border_pad
    )
    fig.add_annotation(
        x=max_x + x_offset, y=y_label,
        text=f"{max_x_label}",
        showarrow=False,
        xanchor="left",
        yanchor=label_yanchor,
        xref="x",
        yref="y",
        textangle=0,
        font=dict(color=font_color, size = 16),
        bgcolor=label_bg_color,
        bordercolor=label_border_color,
        borderpad=label_border_pad
    )

    return fig



def reflect_plot_about_x_axis(fig):
    """
    Reflects all elements of a Plotly figure about x = 0.

    Parameters:
        fig (go.Figure): The figure object containing the plot elements.
    """
    # Reflect traces
    for trace in fig.data:
        if 'x' in trace:
            trace['x'] = [-x for x in trace['x']]  # Reflect x-coordinates

    # Reflect annotations
    new_annotations = []
    for ann in fig.layout.annotations:
        if 'x' in ann:
            ann['x'] = -ann['x']  # Reflect the x-coordinate of the annotation
        new_annotations.append(ann)
    fig.layout.annotations = new_annotations

    # Reflect shapes and lines if any
    if fig.layout.shapes:
        new_shapes = []
        for shape in fig.layout.shapes:
            if 'x0' in shape and 'x1' in shape:
                shape['x0'], shape['x1'] = -shape['x0'], -shape['x1']
            new_shapes.append(shape)
        fig.layout.shapes = new_shapes

    return fig


def reverse_plot_axes(fig, reverse_x=False, reverse_y=False):
    """
    Optionally reverses the x-axis and/or y-axis of a Plotly figure.

    Parameters:
    - fig (go.Figure): The figure object to modify.
    - reverse_x (bool): If True, reverse the x-axis.
    - reverse_y (bool): If True, reverse the y-axis.

    Returns:
    - go.Figure: The modified figure with the specified axes reversed.
    """
    if reverse_x:
        fig.update_layout(xaxis=dict(autorange=False))
    if reverse_y:
        fig.update_layout(yaxis=dict(autorange=False))
    return fig



def add_euler_angles_to_plot(fig, euler_angles, position='top right'):
    """
    Add Euler angles to a Plotly figure as an annotation.

    Parameters:
        fig (go.Figure): The Plotly figure to which the annotation will be added.
        euler_angles (array): The array of Euler angles in degrees.
        position (str): Position for the annotation, default 'top right'.
    """
    # Define positions for the annotation based on the input position argument
    positions = {
        'top right': {'x': 0.95, 'y': 0.95, 'xanchor': 'right', 'yanchor': 'top'},
        'top left': {'x': 0.05, 'y': 0.95, 'xanchor': 'left', 'yanchor': 'top'},
        'bottom left': {'x': 0.05, 'y': 0.05, 'xanchor': 'left', 'yanchor': 'bottom'},
        'bottom right': {'x': 0.95, 'y': 0.05, 'xanchor': 'right', 'yanchor': 'bottom'}
    }

    # Formatting the Euler angles text
    euler_text = f"Euler Angles:<br>X: {euler_angles[0]:.2f}°<br>Y: {euler_angles[1]:.2f}°<br>Z: {euler_angles[2]:.2f}°"

    # Add the text annotation to the plot
    fig.add_annotation(dict(
        text=euler_text,
        x=positions[position]['x'],
        y=positions[position]['y'],
        xanchor=positions[position]['xanchor'],
        yanchor=positions[position]['yanchor'],
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
        ),
        align="left",
        bgcolor="rgba(50, 50, 50, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.2)",
        borderpad=4
    ))

    return fig


def add_euler_angles_to_plot_v2(fig, euler_angles, position='top right'):
    """
    Add Euler angles to a Plotly figure as an annotation, dynamically placed based on current axis ranges.
    """
    # Retrieve axis ranges
    x_range = fig.layout.xaxis.range if fig.layout.xaxis.range else [0, 1]
    y_range = fig.layout.yaxis.range if fig.layout.yaxis.range else [0, 1]

    # Determine normalized positions based on the ranges
    x_pos = x_range[1]  # Use the maximum of the x-range for right side placement
    y_pos = y_range[1]  # Use the maximum of the y-range for top placement

    # Formatting the Euler angles text
    euler_text = f"Euler Angles:<br>X: {euler_angles[0]:.2f}°<br>Y: {euler_angles[1]:.2f}°<br>Z: {euler_angles[2]:.2f}°"

    # Add the text annotation to the plot
    fig.add_annotation(dict(
        text=euler_text,
        x=x_pos, y=y_pos,
        xref="x", yref="y",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
        ),
        align="right",
        bgcolor="rgba(50, 50, 50, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.2)",
        borderpad=4,
        xanchor='right', yanchor='top'  # Anchoring the text right and top
    ))

    return fig


def add_euler_angles_to_plot_v3(fig, euler_angles, euler_convention_str, position='bottom right'):
    """
    Add Euler angles to a Plotly figure as an annotation, dynamically placed based on current axis ranges and specified position.
    """
    # Dictionary to map positions to x, y, xanchor, and yanchor settings
    position_settings = {
        'top right': {'x': 1, 'y': 1, 'xanchor': 'right', 'yanchor': 'top'},
        'top left': {'x': 0, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        'bottom right': {'x': 1, 'y': 0, 'xanchor': 'right', 'yanchor': 'bottom'},
        'bottom left': {'x': 0, 'y': 0, 'xanchor': 'left', 'yanchor': 'bottom'}
    }
    
    # Get the settings based on the position argument
    pos_settings = position_settings.get(position, position_settings['top right'])

        

    # Formatting the Euler angles text
    z_angle = euler_angles[2]
    z_dir = "CW" if z_angle > 0 else ("CCW" if z_angle < 0 else "0")
    z_abs = abs(z_angle)
    euler_text = (
        f"Euler Angles:<br>{euler_convention_str}"
        f"<br>X: {euler_angles[0]:.1f}°"
        f"<br>Y: {euler_angles[1]:.1f}°"
        f"<br>Z: {z_angle:.1f}° ({z_abs:.1f}° {z_dir})"
    )

    # Add the text annotation to the plot
    fig.add_annotation(dict(
        text=euler_text,
        x=pos_settings['x'], y=pos_settings['y'],
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="black"
        ),
        align="left",
        bgcolor="rgba(255, 255, 255, 1)",
        bordercolor="rgba(0, 0, 0, 1)",
        borderpad=4,
        xanchor=pos_settings['xanchor'], yanchor=pos_settings['yanchor']
    ))

    return fig

def add_sagittal_angle_to_plot(fig, euler_z_angle, position='bottom left'):
    """
    Add Euler angles to a Plotly figure as an annotation, dynamically placed based on current axis ranges and specified position.
    """
    # Dictionary to map positions to x, y, xanchor, and yanchor settings
    position_settings = {
        'top right': {'x': 1, 'y': 1, 'xanchor': 'right', 'yanchor': 'top'},
        'top left': {'x': 0, 'y': 1, 'xanchor': 'left', 'yanchor': 'top'},
        'bottom right': {'x': 1, 'y': 0, 'xanchor': 'right', 'yanchor': 'bottom'},
        'bottom left': {'x': 0, 'y': 0, 'xanchor': 'left', 'yanchor': 'bottom'}
    }
    
    # Get the settings based on the position argument
    pos_settings = position_settings.get(position, position_settings['top right'])

    sagittal_angle = euler_z_angle

    # Formatting the Euler angles text
    euler_text = f"Sagittal plane: {sagittal_angle:.1f}°"

    # Add the text annotation to the plot
    fig.add_annotation(dict(
        text=euler_text,
        x=pos_settings['x'], y=pos_settings['y'],
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(
            family="Courier New, monospace",
            size=12,
            color="#ffffff"
        ),
        align="left",
        bgcolor="rgba(50, 50, 50, 0.8)",
        bordercolor="rgba(255, 255, 255, 0.2)",
        borderpad=4,
        xanchor=pos_settings['xanchor'], yanchor=pos_settings['yanchor']
    ))

    return fig


def add_angle_orientation_diagram(fig, position=(0.95, 0.05), size=0.1, text_size=16):
    """
    Adds a diagram to illustrate angle orientation on a Plotly figure.

    Parameters:
    - fig: The Plotly figure to which the diagram will be added.
    - position: A tuple (x, y) for the normalized position of the diagram on the plot.
    - size: The size of the diagram in plot normalized units.
    - text_size: The size of the text for '+' and '-' labels.

    """
    # Define the center of the diagram based on position and size
    center_x, center_y = position
    length = size / 2  # Half size for the line length

    # Add line for 0 degrees
    fig.add_shape(type='line',
                  x0=center_x, y0=center_y - length, x1=center_x, y1=center_y + length,
                  line=dict(color="black", width=2),
                  xref="paper", yref="paper")

    # Add labels for 0 degrees
    fig.add_annotation(x=center_x, y=center_y + length,
                       text="0°", showarrow=False,
                       xref="paper", yref="paper",
                       xanchor="center", yanchor="bottom",
                       font=dict(size=text_size))

    # Add arrow for positive (CW) rotation
    fig.add_shape(type='line',
                  x0=center_x, y0=center_y, x1=center_x + 0.5 * length, y1=center_y + 0.5 * length,
                  line=dict(color="red", width=2, dash='dash'),
                  xref="paper", yref="paper")

    # Label for positive (CW) rotation
    fig.add_annotation(x=center_x + 0.5 * length, y=center_y + 0.5 * length,
                       text="+", showarrow=False,
                       xref="paper", yref="paper",
                       xanchor="left", yanchor="top", font=dict(color="red", size=text_size))

    # Add arrow for negative (CCW) rotation
    fig.add_shape(type='line',
                  x0=center_x, y0=center_y, x1=center_x - 0.5 * length, y1=center_y + 0.5 * length,
                  line=dict(color="blue", width=2, dash='dash'),
                  xref="paper", yref="paper")

    # Label for negative (CCW) rotation
    fig.add_annotation(x=center_x - 0.5 * length, y=center_y + 0.5 * length,
                       text="-", showarrow=False,
                       xref="paper", yref="paper",
                       xanchor="right", yanchor="top", font=dict(color="blue", size=text_size))

    return fig


def add_distance_annotation(fig, 
                            points, 
                            y_position=-0.1, 
                            arrow_color='black', 
                            text_color = "#ffffff", 
                            font_size=12,
                            padding=0.05, 
                            text_box_bg_color = "rgba(50, 50, 50, 0.8)", 
                            text_box_border_color = "rgba(255, 255, 255, 0.2)",
                            x_offset=0.0,
                            start_point=None,
                            end_point=None,
                            segment_offset=(0.0, 0.0),
                            line_width=2,
                            line_dash='solid',
                            show_text=True,
                            show_legend=False,
                            legend_name=None):
    """
    Adds a horizontal arrow annotated with the distance between the minimum and maximum x-values of given points.
    
    Parameters:
        fig (go.Figure): The figure object to modify.
        points (np.array): Array of points [n_points, 2] or [n_points, 3] where x is the first column.
        y_position (float): The y-axis position to place the arrow and text.
        arrow_color (str): Color of the arrow.
        text_color (str): Color of the text.
        font_size (int): Font size of the text annotation.
    """
    # Optional mode: draw arrow along arbitrary segment (start_point -> end_point)
    if start_point is not None and end_point is not None:
        p0 = np.array(start_point, dtype=float) + np.array(segment_offset, dtype=float)
        p1 = np.array(end_point, dtype=float) + np.array(segment_offset, dtype=float)
        dx, dy = p1 - p0
        distance = math.hypot(dx, dy)

        # Main line
        fig.add_trace(go.Scatter(
            x=[p0[0], p1[0]],
            y=[p0[1], p1[1]],
            mode='lines',
            line=dict(color=arrow_color, width=line_width, dash=line_dash),
            showlegend=show_legend,
            name=legend_name if legend_name else None,
            hoverinfo='skip'
        ))

        # Perpendicular end caps at both ends (flat caps)
        if distance > 1e-6:
            ux, uy = dx / distance, dy / distance
            # make caps visibly wide: at least 1.5 mm or 3% of segment length
            perp_scale = max(0.75, 0.03 * distance)
            px, py = -uy * perp_scale, ux * perp_scale
            for (x_base, y_base) in (p0, p1):
                fig.add_shape(type="line",
                              x0=x_base - px, y0=y_base - py,
                              x1=x_base + px, y1=y_base + py,
                              line=dict(color=arrow_color, width=line_width),
                              xref="x", yref="y")

        if show_text and distance is not None:
            mid = 0.5 * (p0 + p1)
            fig.add_annotation(x=mid[0], y=mid[1],
                               text=f"{distance:.1f} mm",
                               showarrow=False,
                               xanchor="center", yanchor="bottom",
                               font=dict(family="Courier New, monospace",
                                         size=font_size,
                                         color=text_color),
                               bgcolor=text_box_bg_color,
                               bordercolor=text_box_border_color,
                               borderpad=4)
        return fig

    # Calculate minimum and maximum x-values
    min_x = np.min(points[:, 0]) + x_offset
    max_x = np.max(points[:, 0]) + x_offset
    distance = max_x - min_x
    
    # Determine the y position at the bottom with padding
    y_range = fig.layout.yaxis.range if fig.layout.yaxis.range else [0, 1]
    y_position = y_range[0] + (y_range[1] - y_range[0]) * padding

    # Draw horizontal line with arrows at both ends
    arrow_size = 0.01 * (max_x - min_x)  # Relative size of arrow heads
    fig.add_shape(type="line",
                  x0=min_x, y0=y_position, x1=max_x, y1=y_position,
                  line=dict(color=arrow_color, width=2),
                  xref="x", yref="y")
    # Left arrowhead
    fig.add_shape(type="line",
                  x0=min_x, y0=y_position, x1=min_x + arrow_size, y1=y_position + arrow_size,
                  line=dict(color=arrow_color, width=2),
                  xref="x", yref="y")
    fig.add_shape(type="line",
                  x0=min_x, y0=y_position, x1=min_x + arrow_size, y1=y_position - arrow_size,
                  line=dict(color=arrow_color, width=2),
                  xref="x", yref="y")
    # Right arrowhead
    fig.add_shape(type="line",
                  x0=max_x, y0=y_position, x1=max_x - arrow_size, y1=y_position + arrow_size,
                  line=dict(color=arrow_color, width=2),
                  xref="x", yref="y")
    fig.add_shape(type="line",
                  x0=max_x, y0=y_position, x1=max_x - arrow_size, y1=y_position - arrow_size,
                  line=dict(color=arrow_color, width=2),
                  xref="x", yref="y")

    # Add a text annotation for the distance in the middle of the arrow
    mid_x = (min_x + max_x) / 2
    fig.add_annotation(x=mid_x, y=y_position,
                        text=f"{distance:.2f} mm",
                        showarrow=False,
                        xanchor="center", yanchor="bottom",
                        font=dict(
                                family="Courier New, monospace",
                                size=font_size,
                                color=text_color),
                        bgcolor=text_box_bg_color,
                        bordercolor=text_box_border_color,
                        borderpad=4
                        )

    return fig

import string

def generate_prostate_template_lattice_dataframe(spacing, range_x, range_y, label1='D', label2='1.5', label2_step=0.5, origin=np.zeros(3), normal_vector=np.array([0,0,1]), edge_vector1=np.array([1,0,0]), edge_vector2=np.array([0,-1,0])):
    # Calculate indices for label1 and label2
    label1_index = string.ascii_uppercase.index(label1.upper()) * 2 + (1 if label1.islower() else 0)
    label2_values = [str(1 + i * label2_step) for i in range(range_y)]
    label2_index = label2_values.index(label2)  # Convert label2 to string if not already

    # Calculate the translation needed to set the special_label point at (0,0,0)
    offset_x = label1_index * spacing * edge_vector1
    offset_y = label2_index * spacing * edge_vector2
    shift_origin = origin - (offset_x + offset_y)

    # Create mesh grid arrays
    x_coords = np.arange(0, range_x) * spacing
    y_coords = np.arange(0, range_y) * spacing

    # Create grid points in 2D
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    x_flat = X.flatten()
    y_flat = Y.flatten()
    points_2d = np.vstack((x_flat, y_flat)).T

    # Convert 2D points into 3D space
    points_3d = points_2d[:, 0][:, np.newaxis] * edge_vector1 + points_2d[:, 1][:, np.newaxis] * edge_vector2 + shift_origin

    # Calculate the rotation matrix to align with the given normal vector
    default_normal = np.array([0, 0, 1])  # Default normal vector
    rotation_matrix, _, _ = calculate_rotation_matrix_and_euler_ZYX_order_tait_bryan_extrinsic(default_normal, normal_vector)
    points_3d = np.dot(points_3d, rotation_matrix.T)

    # Label generation
    labels_1 = [string.ascii_uppercase[i // 2] if i % 2 == 0 else string.ascii_uppercase[i // 2].lower() for i in range(range_x)]
    labels_2 = label2_values * range_x

    # Create DataFrame
    df = pandas.DataFrame(points_3d, columns=['X', 'Y', 'Z'])
    df['Label 1'] = np.repeat(labels_1, range_y)
    df['Label 2'] = labels_2

    return df


def translate_lattice_based_on_labels(df, desired_position_vector, letter_label='D', numerical_label='1.5'):
    """
    Translates the lattice dataframe so that a specified label aligns with the prostate centroid and
    a numerical label aligns 3 units below the highest point of the prostate in the Y direction.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing lattice points with 'X', 'Y', 'Z', 'Label 1', 'Label 2'.
    prostate_centroid (np.array): Coordinates of the prostate centroid (x, y, z).
    max_y_prostate (float): Maximum Y value of the prostate point cloud.
    y_offset (float): Offset in the Y direction for the alignment.
    letter_label (str): The label corresponding to a specific row for the X alignment.
    numerical_label (float): The label corresponding to a specific row for the Y alignment.

    Returns:
    pd.DataFrame: The translated lattice dataframe.
    """
    # Identify the original positions of the labeled point in the lattice
    filtered_df = df[(df['Label 1'] == letter_label) & (df['Label 2'] == numerical_label)]
    if not filtered_df.empty:
        origin_x = filtered_df['X'].values[0]  # Assuming there is at least one match
        origin_y = filtered_df['Y'].values[0]  # Assuming there is at least one match
        origin_z = filtered_df['Z'].values[0]  # Assuming there is at least one match

        # Calculate translation vectors
        translate_x = desired_position_vector[0] - origin_x
        translate_y = desired_position_vector[1] - origin_y
        translate_z = desired_position_vector[2] - origin_z

        # Apply translation
        df['X'] += translate_x
        df['Y'] += translate_y
        df['Z'] += translate_z

    return df



def translate_lattice_in_z(df, min_z_prostate):
    """
    Translates the lattice dataframe in the Z direction so that the entire grid is aligned with the
    most negative Z position of the prostate points array.

    Parameters:
    df (pd.DataFrame): Dataframe containing lattice points with 'X', 'Y', 'Z'.
    min_z_prostate (float): The most negative Z value from the prostate points array.

    Returns:
    pd.DataFrame: The translated lattice dataframe.
    """
    # Find the lowest Z in the lattice
    min_z_lattice = df['Z'].min()

    # Calculate the translation needed in Z
    translate_z = min_z_prostate - min_z_lattice

    # Apply translation
    df['Z'] += translate_z

    return df



def find_nearest_neighbors_sklearn(data_points, query_point, k=3):
    """
    Find the k-nearest neighbors to a query point using sklearn's NearestNeighbors with brute force method.

    Parameters:
    data_points (np.ndarray): Array of data points, shape (N, 3).
    query_point (np.ndarray): Query point, shape (1, 3).
    k (int): Number of nearest neighbors to find.

    Returns:
    np.ndarray: Indices of the k-nearest neighbors.
    """
    # Initialize NearestNeighbors with 'brute' method
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
    nn.fit(data_points)
    # Reshape query_point to fit the expected shape (1, num_features)
    distances, indices = nn.kneighbors(query_point.reshape(1, -1))
    return indices.flatten()


# Function to project points onto a plane and rotate them
def project_and_transform_lines(points, normal, point_on_plane, rotation_matrix, reference_df, line_length=100):
    """
    Project points onto a plane, extend them in the z-direction, and rotate them to match contour plot coordinates.

    Parameters:
    points (np.ndarray): Points to project, shape (N, 3).
    normal (np.ndarray): Normal vector of the plane.
    point_on_plane (np.ndarray): A point on the plane to help define the plane.
    rotation_matrix (np.ndarray): Rotation matrix to align with contour plot.
    reference_df (pd.DataFrame): Dataframe containing additional information for labeling.
    line_length (float): Length of the line to extend in the z-direction.

    Returns:
    list of dicts: Each dict contains 'start', 'end' points of the line and 'label'.
    """
    projected_lines = []
    transformed_lines = []
    nearest_lines_untransformed = []
    for point, label1, label2 in zip(points, reference_df['Label 1'], reference_df['Label 2']):
        # Project point onto plane
        point_to_plane = point - point_on_plane
        projection = point - np.dot(point_to_plane, normal) * normal
        orthogonal_projection_distance = np.linalg.norm(projection-point)
        # Extend in the z-direction
        line_start = projection
        line_end = projection + np.array([0, 0, line_length])
        # Transform points
        transformed_start = np.dot(line_start - point_on_plane, rotation_matrix.T) + point_on_plane
        transformed_end = np.dot(line_end - point_on_plane, rotation_matrix.T) + point_on_plane
        nearest_lines_untransformed.append({
            'start': point,
            'end': point + np.array([0, 0, line_length]),
            'label': f"{label1}-{label2}",
            'orthogonal distance from TRUS plane': orthogonal_projection_distance
        })
        projected_lines.append({
            'start': line_start,
            'end': line_end,
            'label': f"{label1}-{label2}"
        })
        transformed_lines.append({
            'start': transformed_start,
            'end': transformed_end,
            'label': f"{label1}-{label2}",
            'original template line orthogonal distance from TRUS plane': orthogonal_projection_distance
        })

    return transformed_lines, projected_lines, nearest_lines_untransformed



def add_lines_to_contour_plot(fig, lines, line_color='fuchsia', text_position='top center', annotation_x_offset=1.5, annotation_y_shift=12):
    """
    Add lines and annotations to an existing contour plot.

    Parameters:
    fig (go.Figure): The contour plot figure to modify.
    lines (list of dicts): Each dict contains 'start', 'end' points of the line and 'label'.
    line_color (str): Color of the line to draw.
    text_position (str): Position of the text relative to the point.

    Returns:
    go.Figure: The modified figure with lines and annotations added.
    """
    for line in lines:
        # Extract start and end points for the line
        start, end, label = line['start'], line['end'], line['label']

        start_xy = np.array([float(start[2]), float(start[1])], dtype=float)
        end_xy = np.array([float(end[2]), float(end[1])], dtype=float)
        direction = end_xy - start_xy
        direction_norm = np.linalg.norm(direction)
        if direction_norm > 1e-9:
            direction_unit = direction / direction_norm
            # Draw a very long segment; axis clipping makes it appear edge-to-edge.
            extension = 1e4
            draw_start = start_xy - direction_unit * extension
            draw_end = start_xy + direction_unit * extension
        else:
            draw_start = start_xy
            draw_end = end_xy
        
        # Add line trace
        fig.add_trace(go.Scatter(
            x=[draw_start[0], draw_end[0]],  # Axis clipping makes this span the visible plot bounds
            y=[draw_start[1], draw_end[1]],
            mode='lines',
            line=dict(color=line_color, width=4),
            name=label
        ))

        # Add annotation for the line
        fig.add_annotation(
            x=start[2] + annotation_x_offset,
            y=start[1],
            text=label,
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            yshift=annotation_y_shift,
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="black"
            ),
            align="left",
            bgcolor="rgba(255, 255, 255, 1)",
            bordercolor="rgba(0, 0, 0, 1)",
            borderpad=4
        )

    return fig


def create_lines_for_open3d(line_points, colors):
    """
    Create a line set for Open3D visualization from a list of line points and corresponding colors.

    Parameters:
    line_points (list of np.ndarray): List of 2x3 numpy arrays, each representing two points in 3D space.
    colors (list of list): List of colors, each a list of three floats [R, G, B], corresponding to each line.

    Returns:
    o3d.geometry.LineSet: Line set object ready for visualization in Open3D.
    """
    # Initialize the LineSet object
    line_set = o3d.geometry.LineSet()

    # Prepare to aggregate points and lines
    points = []  # List to hold all points
    lines = []   # List to hold line indices
    line_colors = []

    # Process each line segment
    for idx, line_pair in enumerate(line_points):
        # Each line_pair is a 2x3 array; we flatten it into a 6-element array and split it into two points
        start_point = line_pair[0]
        end_point = line_pair[1]

        # Append points
        start_index = len(points)  # Get the current count of points to use as the start index
        end_index = start_index + 1  # The next index is the end index

        points.append(start_point)
        points.append(end_point)

        # Append the line defined by the indices of the start and end points
        lines.append([start_index, end_index])

        # Add the color for this line if provided, else use a default color
        if idx < len(colors):
            line_colors.append(colors[idx])
        else:
            line_colors.append([0, 0, 0])  # Default to black if no color provided

    # Convert lists to numpy arrays for Open3D compatibility
    line_set.points = o3d.utility.Vector3dVector(np.array(points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    return line_set


def create_thick_lines_as_cylinders(line_points, colors, radius=0.5):
    """
    Create a collection of cylindrical meshes approximating thick lines for Open3D visualization.

    Parameters:
    line_points (list of np.ndarray): List of 2x3 numpy arrays, each representing two points in 3D space.
    colors (list of list): List of colors, each a list of three floats [R, G, B], corresponding to each line.
    radii (list of float): Radius of each cylindrical segment corresponding to the lines.

    Returns:
    list of o3d.geometry.TriangleMesh: List of TriangleMesh objects representing cylindrical lines.
    """
    cylinder_meshes = []
    for idx, line_pair in enumerate(line_points):
        # Get start and end points from the line pair
        start_point = line_pair[0]
        end_point = line_pair[1]
        # Calculate the length of the line segment
        length = np.linalg.norm(end_point - start_point)
        # Create a cylinder between the points
        mesh = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=20, split=4)
        # Move cylinder to start point and rotate to align with the line segment
        mesh.translate(-np.array([0, 0, length / 2.0]))  # Center the cylinder at the origin
        direction = (end_point - start_point) / length  # Unit vector in direction of the line
        rotation = mesh.get_rotation_matrix_from_xyz((0, 0, np.arccos(direction[2])))  # Align with z-axis
        mesh.rotate(rotation, center=np.array([0, 0, 0]))
        mesh.translate((start_point + end_point) / 2)  # Move to the midpoint of the segment
        # Color the mesh
        mesh.paint_uniform_color(colors[idx])
        # Append to list of cylinder meshes
        cylinder_meshes.append(mesh)

    return cylinder_meshes

def flatten_geometry_list(geom_list):
    """
    Flattens a list of geometry objects, ensuring it's a flat list of geometries.
    """
    flat_list = []
    for item in geom_list:
        if isinstance(item, list):
            flat_list.extend(item)  # Extend the flat list with the contents of the sublist
        else:
            flat_list.append(item)  # Append the item itself if it's not a list
    return flat_list




def add_transverse_contour_plot_elements(contour_plot, lattice_data, nearest_points, labels):
    """
    Adds lattice points and highlights nearest points on a transverse contour plot.

    Parameters:
    - contour_plot (go.Figure): The existing contour plot to which elements will be added.
    - lattice_data (np.array): Numpy array of shape (N, 2) containing the XY coordinates of the lattice points.
    - nearest_points (np.array): Numpy array of shape (M, 2) containing the XY coordinates of the nearest points.
    - labels (list): List of strings with labels for the nearest points.
    """
    # Add lattice points to the plot
    contour_plot.add_trace(go.Scatter(
        x=lattice_data[:, 0],
        y=lattice_data[:, 1],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Lattice Points'
    ))

    # Add nearest points to the plot
    for point, label in zip(nearest_points, labels):
        contour_plot.add_trace(go.Scatter(
            x=[point[0]],
            y=[point[1]],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[label],
            textposition='top center'
        ))

    return contour_plot




def add_z_angle_line_to_plot(fig, 
                             origin, 
                             euler_angle_z, 
                             length=100):
    """
    Adds a line representing the Z rotation angle to a plotly figure on the XY plane.
    
    Parameters:
    - fig: The Plotly figure to which the line will be added.
    - origin: The origin point for the line (x, y).
    - euler_angle_z: The Z angle in degrees.
    - length: The length of the line.
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(euler_angle_z)

    adjusted_angle_rad = angle_rad - np.pi/2

    if euler_angle_z < 0:
        line_color = 'blue'
    elif euler_angle_z > 0:
        line_color = 'red'
    else:
        line_color = 'black'

    
    # Calculate the endpoint of the line
    end_x = origin[0] + length * np.cos(adjusted_angle_rad)
    end_y = origin[1] + length * np.sin(adjusted_angle_rad)
    
    # Add the line to the figure
    fig.add_trace(go.Scatter(
        x=[origin[0], end_x], 
        y=[origin[1], end_y],
        mode='lines',
        line=dict(color=line_color, width=3, dash='dash'),
        name=f'Sagittal rotation: {euler_angle_z:.1f}°'
    ))

    # Add reference line to the figure
    fig.add_trace(go.Scatter(
        x=[origin[0], origin[0]], 
        y=[origin[1], origin[1] - length],
        mode='lines',
        line=dict(color='black', width=3),
        name=f'Anteroposterior axis (0°)'
    ))

    return fig


def add_perineal_template_lattice_to_transverse_contour_plot(contour_plot, lattice_dataframe, optimal_points, marker_color = 'fuchsia', optimal_marker_accent_color = 'cyan'):
    """
    Add elements to a transverse contour plot for better visualization, including labeled lattice points
    and an emphasized optimal point.
    
    Args:
    - contour_plot: The existing Plotly figure object to update.
    - lattice_dataframe: DataFrame containing coordinates and labels of the lattice points.
    - optimal_point: A tuple or array representing the optimal point to be emphasized.

    Returns:
    - contour_plot: Updated Plotly figure object.
    """
    # Plot each point with labels
    for index, row in lattice_dataframe.iterrows():
        x, y = row['X'], row['Y']
        label = f"{row['Label 1']}-{row['Label 2']}"
        contour_plot.add_trace(go.Scatter(
            x=[x], y=[y], mode='markers+text',
            marker=dict(size=7, color=marker_color),
            text=[label], textposition='top center',
            showlegend=False  # This ensures this trace does not appear in the legend
        ))
    
    # Emphasize the optimal point
    for optimal_point in optimal_points:
        contour_plot.add_trace(go.Scatter(
            x=[optimal_point[0]], y=[optimal_point[1]],
            mode='markers',
            marker=dict(size=15, color=optimal_marker_accent_color, symbol='circle-open'),
            name='Optimal template hole'
        ))

    return contour_plot




def add_perineal_template_lattice_to_transverse_contour_plot_outer_annotations_only(contour_plot,
                                                                                    lattice_dataframe,
                                                                                    optimal_points,
                                                                                    marker_color='fuchsia',
                                                                                    optimal_marker_accent_color='black',
                                                                                    template_label_font_size=12,
                                                                                    optimal_marker_size=18,
                                                                                    optimal_marker_line_width=3):
    """
    Add elements to a transverse contour plot for better visualization, including labeled lattice points
    and an emphasized optimal point.

    Args:
    - contour_plot: The existing Plotly figure object to update.
    - lattice_dataframe: DataFrame containing coordinates and labels of the lattice points.
    - optimal_point: A tuple or array representing the optimal point to be emphasized.

    Returns:
    - contour_plot: Updated Plotly figure object.
    """
    # Determine the min and max values to find the periphery points
    min_x = lattice_dataframe['X'].min()
    max_x = lattice_dataframe['X'].max()
    min_y = lattice_dataframe['Y'].min()
    max_y = lattice_dataframe['Y'].max()

    # Plot each point
    for index, row in lattice_dataframe.iterrows():
        x, y = row['X'], row['Y']
        trace_args = {
            'x': [x],
            'y': [y],
            'mode': 'markers',
            'marker': dict(size=7, color=marker_color),
            'showlegend': False
        }
        contour_plot.add_trace(go.Scatter(**trace_args))

        # Add labels only for points on the periphery
        if x == min_x:
            label = f"{row['Label 2']}"
            trace_args['mode'] = 'markers+text'
            trace_args['text'] = [label.strip()]
            trace_args['textposition'] = 'middle left'
            trace_args['textfont'] = dict(size=template_label_font_size, color='black')
            contour_plot.add_trace(go.Scatter(**trace_args))

        if x == max_x:
            label = f"{row['Label 2']}"
            trace_args['mode'] = 'markers+text'
            trace_args['text'] = [label.strip()]
            trace_args['textposition'] = 'middle right'
            trace_args['textfont'] = dict(size=template_label_font_size, color='black')
            contour_plot.add_trace(go.Scatter(**trace_args))

        if y == max_y:
            label = f" {row['Label 1']}"
            trace_args['mode'] = 'markers+text'
            trace_args['text'] = [label.strip()]
            trace_args['textposition'] = 'bottom center'
            trace_args['textfont'] = dict(size=template_label_font_size, color='black')
            contour_plot.add_trace(go.Scatter(**trace_args))
        
        if y == min_y:
            label = f" {row['Label 1']}"
            trace_args['mode'] = 'markers+text'
            trace_args['text'] = [label.strip()]
            trace_args['textposition'] = 'top center'
            trace_args['textfont'] = dict(size=template_label_font_size, color='black')
            contour_plot.add_trace(go.Scatter(**trace_args))

        

    # Emphasize the optimal point
    for optimal_point in optimal_points:
        contour_plot.add_trace(go.Scatter(
            x=[optimal_point[0]], y=[optimal_point[1]],
            mode='markers',
            marker=dict(size=optimal_marker_size,
                        color=optimal_marker_accent_color,
                        symbol='circle-open',
                        line=dict(color=optimal_marker_accent_color,
                                  width=optimal_marker_line_width)),
            name='Optimal template hole'
        ))

    return contour_plot


def set_square_aspect_ratio(fig):
    """
    Adjusts the aspect ratio of a Plotly figure to be square for contour plots.

    Parameters:
    - fig (plotly.graph_objs.Figure): The figure object to be modified.
    """
    # Set x-axis to anchor to the y-axis
    fig.update_layout(
        xaxis=dict(
            scaleanchor='y',
            scaleratio=1
        ),
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1
        )
    )

    return fig




########### MAIN PLOT CREATOR



def create_advanced_guidance_map_contour_plot(patientUID,
                                            pydicom_item,
                                            dil_ref,
                                            all_ref_key,
                                            oar_ref,
                                            rectum_ref,
                                            structs_referenced_dict,
                                            plot_open3d_structure_set_complete_demonstration_bool,
                                            biopsy_fire_travel_distances,
                                            biopsy_needle_compartment_length,
                                            important_info,
                                            live_display,
                                            transducer_plane_grid_spacing = 2,
                                            prostate_template_spacing = 5, # hole spacing on prostate template
                                            range_x = 13,  # Total points along the first vector
                                            range_y = 13,  # Total points along the second vector
                                            label1 = 'D',  # Corresponding to the first dimension, column to align with prostate centroid
                                            label2 = '1.5',    # Corresponding to the second dimension, row to align with y-shift above max prostate posterior
                                            y_shift_for_1_5_coord_from_prostate_max_post = - 3 # y-shift above max prostate posterior
                                            ):

    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]
    
    # First check that all requisite structures are present
    requisite_structure_types_list = [oar_ref, rectum_ref]
    for row_index, selected_structure_row in sp_patient_selected_structure_info_dataframe.iterrows():
        selected_structure_info = selected_structure_row.to_dict()
        selected_structure_ID = selected_structure_info["Structure ID"]
        selected_structure_ref_type = selected_structure_info["Struct ref type"]
        selected_structure_ref_num = selected_structure_info["Dicom ref num"]
        selected_structure_structure_index = selected_structure_info["Index number"]
        structure_found_bool = selected_structure_info["Struct found bool"]
        if (structure_found_bool == False) and (selected_structure_ref_type in requisite_structure_types_list):
            # return empty figure if the requisite structures are not present
            important_info.add_text_line("Creating production plots.", live_display)
            empty_fig = go.Figure()
            return empty_fig


    non_dil_list_of_pcds = []

    biopsy_needle_tip_length = 6

    #relative_dil_index = 2
    #patient_uid = '181_F2 ()'

    #grid_spacing = 2

    entire_lattice_df = pydicom_item[all_ref_key]['Multi-structure information dict (not for csv output)']['Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe']

    ### pcd 
    # entire_lattice_pcd = point_containment_tools.create_point_cloud(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
    #                                   'Test location (Prostate centroid origin) (Y)',
    #                                   'Test location (Prostate centroid origin) (Z)']].to_numpy())
    entire_lattice_pcd_colored = dataframe_to_point_cloud(entire_lattice_df, 'Test location (Prostate centroid origin) (X)',
                                      'Test location (Prostate centroid origin) (Y)',
                                      'Test location (Prostate centroid origin) (Z)')
    non_dil_list_of_pcds.append(entire_lattice_pcd_colored)
    ###


    specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
    selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

    selected_prostate_ID = selected_prostate_info["Structure ID"]
    selected_prostate_ref_type = selected_prostate_info["Struct ref type"]
    selected_prostate_ref_num = selected_prostate_info["Dicom ref num"]
    selected_prostate_structure_index = selected_prostate_info["Index number"]
    prostate_found_bool = selected_prostate_info["Struct found bool"]

    ### pcd 
    origin_pcd = point_containment_tools.create_point_cloud(np.array([[0,0,0]]), color = np.array([1,0,1]))
    non_dil_list_of_pcds.append(origin_pcd)
    axes_line_set = plotting_funcs.create_colored_origin_axes_o3d_lineset()
    non_dil_list_of_pcds.append(axes_line_set)

     

    prostate_grid_template_lattice_dataframe = generate_prostate_template_lattice_dataframe(prostate_template_spacing, 
                                                                                            range_x, 
                                                                                            range_y, 
                                                                                            label1, 
                                                                                            label2)
    
    #prostate_grid_template_lattice_arr = prostate_grid_template_lattice_dataframe[['X','Y','Z']].to_numpy()
    
    #pcd
    # prostate_grid_template_lattice_pcd = point_containment_tools.create_point_cloud(prostate_grid_template_lattice_arr, color = np.array([0,1,0]))
    # non_dil_list_of_pcds.append(prostate_grid_template_lattice_pcd)

    # plotting_funcs.plot_geometries(prostate_grid_template_lattice_pcd,
    #                         axes_line_set
    #                         )


    prostate_centroid = pydicom_item[oar_ref][selected_prostate_structure_index]["Structure global centroid"].reshape(3)
    prostate_inter_slice_interp_np_arr = pydicom_item[oar_ref][selected_prostate_structure_index]['Inter-slice interpolation information'].interpolated_pts_np_arr
    prostate_inter_slice_interp_np_arr_prostate_coords = prostate_inter_slice_interp_np_arr - prostate_centroid
    max_y_prostate = np.max(prostate_inter_slice_interp_np_arr_prostate_coords[:, 1])

    # pcd 
    pcd_color_arr = structs_referenced_dict[selected_prostate_ref_type]['PCD color']
    prostate_inter_slice_interp_prostate_coords_pcd = point_containment_tools.create_point_cloud(prostate_inter_slice_interp_np_arr_prostate_coords, color = pcd_color_arr)
    non_dil_list_of_pcds.append(prostate_inter_slice_interp_prostate_coords_pcd)


    # prostate trimesh 
    prostate_trimesh = pydicom_item[oar_ref][selected_prostate_structure_index]['Structure OPEN3D triangle mesh object']
    prostate_trimesh_prostate_frame = translate_mesh(prostate_trimesh, -prostate_centroid)

    
    structure_trimesh_objs_list_of_dicts = [{'Structure ID': selected_prostate_ID, 'Structure trimesh': prostate_trimesh_prostate_frame, 'Color': pcd_color_arr, 'Tag': 'Prostate'}]
    for row_index, selected_structure_row in sp_patient_selected_structure_info_dataframe.iterrows():

        selected_structure_info = selected_structure_row.to_dict()

        selected_structure_ID = selected_structure_info["Structure ID"]
        selected_structure_ref_type = selected_structure_info["Struct ref type"]
        selected_structure_ref_num = selected_structure_info["Dicom ref num"]
        selected_structure_structure_index = selected_structure_info["Index number"]
        structure_found_bool = selected_structure_info["Struct found bool"]
        
        # pcd 
        pcd_color_arr = structs_referenced_dict[selected_structure_ref_type]['PCD color']

        if structure_found_bool == True:
            structure_inter_slice_interp_np_arr = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Inter-slice interpolation information'].interpolated_pts_np_arr
            structure_inter_slice_interp_np_arr_prostate_coords = structure_inter_slice_interp_np_arr - prostate_centroid
            
            # pcd 
            structure_inter_slice_interp_prostate_coords_pcd = point_containment_tools.create_point_cloud(structure_inter_slice_interp_np_arr_prostate_coords, color = np.array([0,1,1]))
            non_dil_list_of_pcds.append(structure_inter_slice_interp_prostate_coords_pcd)

            # trimesh
            structure_trimesh = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Structure OPEN3D triangle mesh object']
            structure_trimesh_prostate_frame = translate_mesh(structure_trimesh, -prostate_centroid)
        else: 
            pass
        
        if selected_structure_ref_type == rectum_ref:

            point1 = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Structure centroid pts'][0]
            point2 = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Structure centroid pts'][-1]
            rectum_point_1_prostate_frame = point1 - prostate_centroid
            rectum_point_2_prostate_frame = point2 - prostate_centroid

            rectum_point_1_prostate_frame_pcd = point_containment_tools.create_point_cloud(rectum_point_1_prostate_frame.reshape(1, -1), color = np.array([1,0,0]))
            rectum_point_2_prostate_frame_pcd = point_containment_tools.create_point_cloud(rectum_point_2_prostate_frame.reshape(1, -1), color = np.array([1,0,0]))
            non_dil_list_of_pcds.append(rectum_point_1_prostate_frame_pcd)
            non_dil_list_of_pcds.append(rectum_point_2_prostate_frame_pcd)

            min_z_rectum = np.min(structure_inter_slice_interp_np_arr_prostate_coords[:, 2])
            structure_trimesh_objs_list_of_dicts.append({'Structure ID': selected_structure_ID, 'Structure trimesh': structure_trimesh_prostate_frame, 'Color': pcd_color_arr, 'Tag': 'Rectum'})

        else:
            structure_trimesh_objs_list_of_dicts.append({'Structure ID': selected_structure_ID, 'Structure trimesh': structure_trimesh_prostate_frame, 'Color': pcd_color_arr, 'Tag': None})

            


    # Assuming prostate_grid_template_lattice_dataframe, prostate_centroid, and prostate_points are already defined:
    coord_1_5_pos_in_prostate_coords = max_y_prostate + y_shift_for_1_5_coord_from_prostate_max_post
    grid_position_D_1_5_in_prostate_coords = np.array([0,coord_1_5_pos_in_prostate_coords,min_z_rectum])
    
    # Translate the lattice dataframe based on specified labels
    prostate_grid_template_lattice_XYZ_aligned_dataframe = translate_lattice_based_on_labels(prostate_grid_template_lattice_dataframe, 
                                                                                            grid_position_D_1_5_in_prostate_coords, 
                                                                                            letter_label=label1, 
                                                                                            numerical_label=label2)
    # Further translate the lattice in Z
    #prostate_grid_template_lattice_XYZ_aligned_dataframe = translate_lattice_in_z(prostate_grid_template_lattice_XY_aligned_dataframe, min_z_prostate)

    prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr = prostate_grid_template_lattice_XYZ_aligned_dataframe[['X','Y','Z']].to_numpy()
    #prostate_grid_template_translation_distance = prostate_inter_slice_interp_np_arr_prostate_coords[:,2].min()
    #prostate_grid_template_lattice_prostate_coord_frame_arr = prostate_grid_template_lattice_arr + np.array([0,0,prostate_grid_template_translation_distance])
    
    #pcd
    prostate_grid_template_lattice_prostate_coord_frame_pcd = point_containment_tools.create_point_cloud(prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr, color = np.array([0,1,0]))
    non_dil_list_of_pcds.append(prostate_grid_template_lattice_prostate_coord_frame_pcd)

    contour_plot_list_of_dicts = []
    dil_specific_pointclouds_list = []
    for specific_dil_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        optimal_positions_df = specific_dil_structure['Biopsy optimization: Optimal biopsy location dataframe']
        dil_id_from_pydicom = specific_dil_structure['ROI']
        # Extract the base point from optimal_positions_df using the first matched index
        #sp_dil_optimal_positions_df = optimal_positions_df[optimal_positions_df['Relative DIL index'] == specific_dil_index]
        sp_dil_optimal_coordinate = np.array([
            optimal_positions_df.at[0, 'Test location (Prostate centroid origin) (X)'],
            optimal_positions_df.at[0, 'Test location (Prostate centroid origin) (Y)'],
            optimal_positions_df.at[0, 'Test location (Prostate centroid origin) (Z)']
        ])
        sp_dil_id = optimal_positions_df.at[0, 'Relative DIL ID']

        dil_inter_slice_interp_np_arr = specific_dil_structure['Inter-slice interpolation information'].interpolated_pts_np_arr
        dil_inter_slice_interp_np_arr_prostate_coords = dil_inter_slice_interp_np_arr - prostate_centroid
        
        #pcd 
        dil_inter_slice_interp_prostate_coords_pcd = point_containment_tools.create_point_cloud(dil_inter_slice_interp_np_arr_prostate_coords, color = np.array([0,0,1]))
        optimal_pos_pcd = point_containment_tools.create_point_cloud(sp_dil_optimal_coordinate.reshape(1, -1), color = np.array([0,1,0]))
        dil_specific_pointclouds_list.append(dil_inter_slice_interp_prostate_coords_pcd)
        dil_specific_pointclouds_list.append(optimal_pos_pcd)


        transducer_plane_df, transducer_plane_normal, euler_angles, euler_convention_str = generate_grid_dataframe(rectum_point_1_prostate_frame, 
                                                                                             rectum_point_2_prostate_frame, 
                                                                                             transducer_plane_grid_spacing, 
                                                                                             entire_lattice_df, 
                                                                                             sp_dil_optimal_coordinate)
        transformed_grid_df, transformed_optimal_point, rotation_matrix = transform_grid_and_point_for_plotting(transducer_plane_df, transducer_plane_normal, sp_dil_optimal_coordinate)

        # pcd 
        transducer_plane_pcd = point_containment_tools.create_point_cloud(transducer_plane_df[['Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)']].to_numpy(), color = np.array([0,0,1]))
        #transducer_plane_rotated_pcd = point_containment_tools.create_point_cloud(transformed_grid_df[['Transformed X', 'Transformed Y', 'Transformed Z']].to_numpy(), color = np.array([0,0,1]))
        #transformed_optimal_point_pcd = point_containment_tools.create_point_cloud(transformed_optimal_point.reshape(1, -1), color = np.array([1,0,0]))
        #transducer_plane_pcd_colored = dataframe_to_point_cloud(transducer_plane_df, 'Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)')
        dil_specific_pointclouds_list.append(transducer_plane_pcd)

        # Using the function to find the nearest neighbors
        prostate_template_points = prostate_grid_template_lattice_XYZ_aligned_dataframe[['X', 'Y', 'Z']].values
        nearest_indices = find_nearest_neighbors_sklearn(prostate_template_points, sp_dil_optimal_coordinate, k=1)
        nearest_template_points = prostate_template_points[nearest_indices]


        # Project, transform, and annotate lines
        nearest_prostate_template_lines_contour_plot_coords_list_of_dicts, nearest_prostate_template_lines_transducer_plane_projected_list_of_dicts, nearest_prostate_template_lines_list_of_dicts = project_and_transform_lines(nearest_template_points, 
                                            transducer_plane_normal, 
                                            sp_dil_optimal_coordinate, 
                                            rotation_matrix, 
                                            prostate_grid_template_lattice_XYZ_aligned_dataframe.iloc[nearest_indices])


        line_pts_list = [np.array([item['start'],item['end']]) for item in nearest_prostate_template_lines_list_of_dicts]
        nearest_template_lines_colors_list = [np.array([1,0,1]) for i in range(len(line_pts_list))]
        #nearest_template_lines_lineset = create_lines_for_open3d(line_pts_list, nearest_template_lines_colors_list)
        nearest_template_lines_cylinder_mesh_list = create_thick_lines_as_cylinders(line_pts_list, nearest_template_lines_colors_list, radius=0.25)
        #nearest_template_lines_cylinder_mesh_flat_list = flatten_geometry_list(nearest_template_lines_cylinder_mesh_list)
        dil_specific_pointclouds_list.extend(nearest_template_lines_cylinder_mesh_list)

        # dil trimesh
        dil_trimesh = specific_dil_structure['Structure OPEN3D triangle mesh object']
        dil_trimesh_prostate_frame = translate_mesh(dil_trimesh, -prostate_centroid)

        if plot_open3d_structure_set_complete_demonstration_bool == True:
            prostate_frame_representation_pcd_list = non_dil_list_of_pcds + dil_specific_pointclouds_list
            plotting_funcs.plot_geometries(*prostate_frame_representation_pcd_list)

        

        # create plot
        contour_plot = plot_transformed_contour(transformed_grid_df)

        contour_plot.update_layout(
                title=dict(
                    text=f"TRUS plane guidance map - {patientUID} - {sp_dil_id}"
                )
            )


        # prostate_mesh_slice_pts = slice_mesh_fast(prostate_trimesh_prostate_frame, transducer_plane_normal, sp_dil_optimal_coordinate)
        # prostate_mesh_slice_pts_transformed = transform_points(prostate_mesh_slice_pts, rotation_matrix, transducer_plane_df)
        # prostate_mesh_slice_pts_transformed_contour_plot_coords = prostate_mesh_slice_pts_transformed[:,[2,1]]
        # #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
        # contour_plot = plot_transformed_mesh_slice(contour_plot, prostate_mesh_slice_pts_transformed_contour_plot_coords, 'Prostate', color_input = 'red')
        
        rectum_plus_prostate_pts_contour_plot_coords_list = []
        for trimesh_dict in structure_trimesh_objs_list_of_dicts:
            contour_color = trimesh_dict['Color']
            structure_trimesh = trimesh_dict['Structure trimesh']
            structure_ID = trimesh_dict['Structure ID']
            tag = trimesh_dict['Tag']

            structure_mesh_slice_pts = slice_mesh_fast(structure_trimesh, transducer_plane_normal, sp_dil_optimal_coordinate)
            structure_mesh_slice_pts_transformed = transform_points(structure_mesh_slice_pts, rotation_matrix, transducer_plane_df)
            structure_mesh_slice_pts_transformed_contour_plot_coords = structure_mesh_slice_pts_transformed[:,[2,1]]
            #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
            contour_plot = plot_transformed_mesh_slice(contour_plot, 
                                                       structure_mesh_slice_pts_transformed_contour_plot_coords, 
                                                       structure_ID, 
                                                       color_input = misc_tools.normalize_color_values(contour_color))
            
            if tag == 'Prostate' or tag == 'Rectum':
                rectum_plus_prostate_pts_contour_plot_coords_list.append(structure_mesh_slice_pts_transformed_contour_plot_coords)
            if tag == 'Prostate':
                prostate_mesh_slice_pts_transformed_contour_plot_coords = structure_mesh_slice_pts_transformed_contour_plot_coords

        dil_mesh_slice_pts = slice_mesh_fast(dil_trimesh_prostate_frame, transducer_plane_normal, sp_dil_optimal_coordinate)
        dil_mesh_slice_pts_transformed = transform_points(dil_mesh_slice_pts, rotation_matrix, transducer_plane_df)
        dil_mesh_slice_pts_transformed_contour_plot_coords = dil_mesh_slice_pts_transformed[:,[2,1]]
        #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
        contour_plot = plot_transformed_mesh_slice(contour_plot, dil_mesh_slice_pts_transformed_contour_plot_coords, sp_dil_id, color_input = 'green')


        #contour_plot = reflect_plot_about_x_axis(contour_plot) 

        #prostate_mesh_slice_pts_transformed_contour_plot_coords_reflected = copy.deepcopy(prostate_mesh_slice_pts_transformed_contour_plot_coords)
        #prostate_mesh_slice_pts_transformed_contour_plot_coords_reflected[:,0] = -prostate_mesh_slice_pts_transformed_contour_plot_coords_reflected[:,0]

        rectum_plus_prostate_pts_contour_plot_coords = np.vstack(rectum_plus_prostate_pts_contour_plot_coords_list)
        #rectum_plus_prostate_pts_contour_plot_coords_reflected = copy.deepcopy(rectum_plus_prostate_pts_contour_plot_coords)
        #rectum_plus_prostate_pts_contour_plot_coords_reflected[:,0] = -rectum_plus_prostate_pts_contour_plot_coords_reflected[:,0]


        #contour_plot = reverse_plot_axes(contour_plot, reverse_x=True, reverse_y=True)

        contour_plot = adjust_plot_area_and_reverse_axes(contour_plot, rectum_plus_prostate_pts_contour_plot_coords, margin=5, reverse_x=True, reverse_y=True)

        transformed_optimal_point_contour_coord_sys = transformed_optimal_point[[2,1]]
        contour_plot = add_points_to_plot_v2(contour_plot, 
                                     transformed_optimal_point_contour_coord_sys, 
                                     f"Optimal ({sp_dil_id}) | [{transformed_optimal_point_contour_coord_sys[0]:.2f}, {transformed_optimal_point_contour_coord_sys[1]:.2f}] mm", 
                                     legend_name = f"Optimal ({sp_dil_id})")

        custom_height = 0
        custom_height_step = 2 
        for index,penetration_depth in enumerate(biopsy_fire_travel_distances):
            custom_height = custom_height + custom_height_step*(index+1)
            needle_tip_position_before_firing = copy.deepcopy(transformed_optimal_point_contour_coord_sys)
            needle_tip_position_before_firing[0] = needle_tip_position_before_firing[0] - penetration_depth + (biopsy_needle_tip_length + biopsy_needle_compartment_length/2)
            contour_plot = add_points_to_plot_v2(contour_plot, 
                                                needle_tip_position_before_firing, 
                                                f"Nx tip fire position (Pene. dep.: {penetration_depth:.1f} mm) | [{needle_tip_position_before_firing[0]:.2f}, {needle_tip_position_before_firing[1]:.2f}] mm", 
                                                color = 'red', 
                                                symbol = 'arrow-bar-left', 
                                                custom_height = custom_height, 
                                                legend_name = "Fire pos. ("+str(penetration_depth)+" mm)",
                                                color_index = index)


        contour_plot = add_lines_to_contour_plot(contour_plot, nearest_prostate_template_lines_contour_plot_coords_list_of_dicts)


        contour_plot = add_x_bounds_with_annotations(contour_plot, 
                                        prostate_mesh_slice_pts_transformed_contour_plot_coords, 
                                        y_position = -0.075, 
                                        max_x_label="Prostate base", 
                                        min_x_label="Prostate apex", 
                                        line_color_max='black', 
                                        line_color_min='black', 
                                        line_style_max='dot',
                                        line_style_min='dot', 
                                        line_width=3 
                                        )


        contour_plot = add_distance_annotation(contour_plot, 
                                               prostate_mesh_slice_pts_transformed_contour_plot_coords, 
                                               y_position=-0.075, 
                                               arrow_color='black')

        contour_plot = add_euler_angles_to_plot_v3(contour_plot, 
                                                   euler_angles,
                                                   euler_convention_str,
                                                   position='bottom right')

        sp_dil_contour_plot_dict = {"DIL ID": dil_id_from_pydicom,
                                    "Contour plot": contour_plot}
        contour_plot_list_of_dicts.append(sp_dil_contour_plot_dict)

    return contour_plot_list_of_dicts
    





def create_advanced_guidance_map_transducer_saggital_and_transverse_contour_plot(patientUID,
                                            pydicom_item,
                                            dil_ref,
                                            all_ref_key,
                                            oar_ref,
                                            rectum_ref,
                                            structs_referenced_dict,
                                            plot_open3d_structure_set_complete_demonstration_bool,
                                            biopsy_fire_travel_distances,
                                            biopsy_needle_compartment_length,
                                            interp_inter_slice_dist,
                                            interp_intra_slice_dist,
                                            radius_for_normals_estimation,
                                            max_nn_for_normals_estimation,
                                            important_info,
                                            live_display,
                                            biopsy_needle_tip_length,
                                            transducer_plane_grid_spacing = 2,
                                            prostate_template_spacing = 5, # hole spacing on prostate template
                                            range_x = 13,  # Total points along the first vector
                                            range_y = 13,  # Total points along the second vector
                                            label1 = 'D',  # Corresponding to the first dimension, column to align with prostate centroid
                                            label2 = '1.5',    # Corresponding to the second dimension, row to align with y-shift above max prostate posterior
                                            y_shift_for_1_5_coord_from_prostate_max_post = - 3, # y-shift above max prostate posterior
                                            simple_angle_display_option_bool = False,
                                            use_natural_TRUS_origin_for_transducer_sagittal_plane = True, # if False, will use rectum structure centroids
                                            template_label_font_size = 12,
                                            draw_orientation_diagram = True,
                                            colorbar_title_font_size = 12,
                                            fire_annotation_style = "hockey",
                                            fire_table_position = "auto"
                                            ):

    def _anchor_colorbars_bottom_right(fig):
        """Place contour colorbars bottom-right outside plot area with left-side vertical labels."""
        colorbar_x = 1.13
        colorbar_len = 0.5
        colorbar_y = 0.0
        label_x_offset = 0.012
        for tr in fig.data:
            if getattr(tr, "type", None) == "contour" and hasattr(tr, "colorbar") and tr.colorbar:
                title_text = ""
                title_font_size = 12
                if getattr(tr.colorbar, "title", None) is not None:
                    if getattr(tr.colorbar.title, "text", None):
                        title_text = tr.colorbar.title.text
                    if getattr(tr.colorbar.title, "font", None) is not None and getattr(tr.colorbar.title.font, "size", None):
                        title_font_size = tr.colorbar.title.font.size

                tr.update(colorbar=dict(
                    x=colorbar_x,
                    xanchor="left",
                    y=colorbar_y,
                    yanchor="bottom",
                    lenmode="fraction",
                    len=colorbar_len,
                    orientation="v",
                    title=dict(text="")
                ))

                if title_text:
                    fig.add_annotation(
                        x=colorbar_x - label_x_offset,
                        y=colorbar_y + 0.5 * colorbar_len,
                        xref="paper",
                        yref="paper",
                        text=title_text,
                        textangle=-90,
                        showarrow=False,
                        xanchor="center",
                        yanchor="middle",
                        font=dict(size=title_font_size, color="black")
                    )

    if fire_annotation_style not in ["hockey", "compact_table"]:
        warnings.warn(f"Unsupported fire_annotation_style '{fire_annotation_style}'. Falling back to 'hockey'.")
        fire_annotation_style = "hockey"

    valid_fire_table_positions = [
        "auto",
        "outside top center",
        "top left",
        "top right",
        "bottom left",
        "bottom right",
        "middle left",
        "middle right",
        "middle top",
        "middle bottom",
    ]
    fire_table_position = (fire_table_position or "auto").lower()
    if fire_table_position not in valid_fire_table_positions:
        warnings.warn(f"Unsupported fire_table_position '{fire_table_position}'. Falling back to 'auto'.")
        fire_table_position = "auto"

    def _show_all_axis_lines(fig):
        """Turn on all four axis lines with ticks/labels on all sides (primary + overlay axes), including minor ticks/grid."""
        def _ensure_overlay_axis_anchor():
            has_overlay_trace = any(
                (getattr(trace, "xaxis", None) == "x2" and getattr(trace, "yaxis", None) == "y2")
                for trace in fig.data
            )
            if has_overlay_trace:
                return

            x_range = fig.layout.xaxis.range if getattr(fig.layout, "xaxis", None) and fig.layout.xaxis.range else [0, 1]
            y_range = fig.layout.yaxis.range if getattr(fig.layout, "yaxis", None) and fig.layout.yaxis.range else [0, 1]
            x_mid = 0.5 * (x_range[0] + x_range[1])
            y_mid = 0.5 * (y_range[0] + y_range[1])
            fig.add_trace(go.Scatter(
                x=[x_mid],
                y=[y_mid],
                xaxis="x2",
                yaxis="y2",
                mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                showlegend=False,
                hoverinfo="skip",
                name="_overlay_axis_anchor"
            ))

        minor_axis = dict(
            ticks="inside",
            ticklen=4,
            tickcolor="black",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            gridwidth=0.5
        )
        common_axis = dict(
            showline=True,
            ticks="inside",
            showticklabels=True,
            automargin=True,
            linewidth=2,
            linecolor="black",
            tickcolor="black",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
            gridwidth=1,
            zeroline=True,
            zerolinecolor="rgba(55,55,55,1)",
            zerolinewidth=2,
            minor=minor_axis
        )
        fig.update_layout(
            xaxis=dict(common_axis, side="bottom", ticklabelposition="outside", mirror="allticks"),
            yaxis=dict(common_axis, side="left", ticklabelposition="outside", mirror="allticks"),
            xaxis2=dict(common_axis,
                        overlaying="x",
                        anchor="y",
                        position=1,
                        side="top",
                        matches="x",
                        showticklabels=True,
                        ticklabelposition="outside",
                        showspikes=False),
            yaxis2=dict(common_axis,
                        overlaying="y",
                        anchor="x",
                        position=1,
                        side="right",
                        matches="y",
                        showticklabels=True,
                        ticklabelposition="outside",
                        showspikes=False)
        )
        # Re-apply to all axes to ensure top/bottom and left/right all render minor ticks.
        fig.update_xaxes(minor=minor_axis, ticks="inside", showline=True, showticklabels=True)
        fig.update_yaxes(minor=minor_axis, ticks="inside", showline=True, showticklabels=True)
        _ensure_overlay_axis_anchor()


    sp_patient_selected_structure_info_dataframe = pydicom_item[all_ref_key]["Multi-structure pre-processing output dataframes dict"]["Selected structures"]
    
    # First check that all requisite structures are present
    requisite_structure_types_list = [oar_ref, rectum_ref]
    for row_index, selected_structure_row in sp_patient_selected_structure_info_dataframe.iterrows():
        selected_structure_info = selected_structure_row.to_dict()
        selected_structure_ID = selected_structure_info["Structure ID"]
        selected_structure_ref_type = selected_structure_info["Struct ref type"]
        selected_structure_ref_num = selected_structure_info["Dicom ref num"]
        selected_structure_structure_index = selected_structure_info["Index number"]
        structure_found_bool = selected_structure_info["Struct found bool"]
        if (structure_found_bool == False) and (selected_structure_ref_type in requisite_structure_types_list):
            # return empty figure if the requisite structures are not present
            important_info.add_text_line("Creating production plots.", live_display)
            empty_fig = go.Figure()
            return empty_fig


    non_dil_list_of_pcds = []


    #relative_dil_index = 2
    #patient_uid = '181_F2 ()'

    #grid_spacing = 2

    entire_lattice_df = pydicom_item[all_ref_key]['Multi-structure information dict (not for csv output)']['Biopsy optimization: Optimal biopsy location (entire cubic lattice) dataframe']

    ### pcd 
    # entire_lattice_pcd = point_containment_tools.create_point_cloud(entire_lattice_df[['Test location (Prostate centroid origin) (X)',
    #                                   'Test location (Prostate centroid origin) (Y)',
    #                                   'Test location (Prostate centroid origin) (Z)']].to_numpy())
    entire_lattice_pcd_colored = dataframe_to_point_cloud(entire_lattice_df, 'Test location (Prostate centroid origin) (X)',
                                      'Test location (Prostate centroid origin) (Y)',
                                      'Test location (Prostate centroid origin) (Z)',
                                      filter_below_threshold=True,
                                      threshold=0.05)
    non_dil_list_of_pcds.append(entire_lattice_pcd_colored)
    ###


    specific_prostate_info_df = sp_patient_selected_structure_info_dataframe[sp_patient_selected_structure_info_dataframe["Struct ref type"] == oar_ref]
    selected_prostate_info = specific_prostate_info_df.to_dict('records')[0]

    selected_prostate_ID = selected_prostate_info["Structure ID"]
    selected_prostate_ref_type = selected_prostate_info["Struct ref type"]
    selected_prostate_ref_num = selected_prostate_info["Dicom ref num"]
    selected_prostate_structure_index = selected_prostate_info["Index number"]
    prostate_found_bool = selected_prostate_info["Struct found bool"]

    ### pcd 
    origin_pcd = point_containment_tools.create_point_cloud(np.array([[0,0,0]]), color = np.array([1,0,1]))
    non_dil_list_of_pcds.append(origin_pcd)
    axes_line_set = plotting_funcs.create_colored_origin_axes_o3d_lineset()
    non_dil_list_of_pcds.append(axes_line_set)

     

    prostate_grid_template_lattice_dataframe = generate_prostate_template_lattice_dataframe(prostate_template_spacing, 
                                                                                            range_x, 
                                                                                            range_y, 
                                                                                            label1, 
                                                                                            label2)
    
    #prostate_grid_template_lattice_arr = prostate_grid_template_lattice_dataframe[['X','Y','Z']].to_numpy()
    
    #pcd
    # prostate_grid_template_lattice_pcd = point_containment_tools.create_point_cloud(prostate_grid_template_lattice_arr, color = np.array([0,1,0]))
    # non_dil_list_of_pcds.append(prostate_grid_template_lattice_pcd)

    # plotting_funcs.plot_geometries(prostate_grid_template_lattice_pcd,
    #                         axes_line_set
    #                         )


    prostate_centroid = pydicom_item[oar_ref][selected_prostate_structure_index]["Structure global centroid"].reshape(3)
    prostate_inter_slice_interp_np_arr = pydicom_item[oar_ref][selected_prostate_structure_index]['Inter-slice interpolation information'].interpolated_pts_np_arr
    prostate_inter_slice_interp_np_arr_prostate_coords = prostate_inter_slice_interp_np_arr - prostate_centroid
    max_y_prostate = np.max(prostate_inter_slice_interp_np_arr_prostate_coords[:, 1])
    min_z_prostate = np.min(prostate_inter_slice_interp_np_arr_prostate_coords[:, 2])


    # pcd 
    # pcd_color_arr = structs_referenced_dict[selected_prostate_ref_type]['PCD color']
    # prostate_inter_slice_interp_prostate_coords_pcd = point_containment_tools.create_point_cloud(prostate_inter_slice_interp_np_arr_prostate_coords, color = pcd_color_arr)
    # non_dil_list_of_pcds.append(prostate_inter_slice_interp_prostate_coords_pcd)


    # prostate trimesh 
    #prostate_trimesh = pydicom_item[oar_ref][selected_prostate_structure_index]['Structure OPEN3D triangle mesh object']
    #prostate_interpolation_information = pydicom_item[oar_ref][selected_prostate_structure_index]["Intra-slice interpolation information"]
    #prostate_threeDdata_array_fully_interpolated_with_end_caps = prostate_interpolation_information.interpolated_pts_with_end_caps_np_arr
    # prostate_trimesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
    #                         interp_intra_slice_dist,
    #                         prostate_threeDdata_array_fully_interpolated_with_end_caps,
    #                         radius_for_normals_estimation,
    #                         max_nn_for_normals_estimation
    #                         )
    # prostate_trimesh_prostate_frame = translate_mesh(prostate_trimesh, -prostate_centroid)
    # structure_trimesh_objs_list_of_dicts = [{'Structure ID': selected_prostate_ID, 'Structure trimesh': prostate_trimesh_prostate_frame, 'Color': pcd_color_arr, 'Tag': 'Prostate'}]
    structure_trimesh_objs_list_of_dicts = []
    for row_index, selected_structure_row in sp_patient_selected_structure_info_dataframe.iterrows():

        selected_structure_info = selected_structure_row.to_dict()

        selected_structure_ID = selected_structure_info["Structure ID"]
        selected_structure_ref_type = selected_structure_info["Struct ref type"]
        selected_structure_ref_num = selected_structure_info["Dicom ref num"]
        selected_structure_structure_index = selected_structure_info["Index number"]
        structure_found_bool = selected_structure_info["Struct found bool"]
        
        # pcd 
        pcd_color_arr = structs_referenced_dict[selected_structure_ref_type]['PCD color']

        if structure_found_bool == True:
            structure_inter_slice_interp_np_arr = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Inter-slice interpolation information'].interpolated_pts_np_arr
            structure_inter_slice_interp_np_arr_prostate_coords = structure_inter_slice_interp_np_arr - prostate_centroid
            
            # pcd 
            structure_inter_slice_interp_prostate_coords_pcd = point_containment_tools.create_point_cloud(structure_inter_slice_interp_np_arr_prostate_coords, color = pcd_color_arr)
            non_dil_list_of_pcds.append(structure_inter_slice_interp_prostate_coords_pcd)

            # trimesh
            #structure_trimesh = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Structure OPEN3D triangle mesh object']
            structure_interpolation_information = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]["Intra-slice interpolation information"]
            structure_threeDdata_array_fully_interpolated_with_end_caps = structure_interpolation_information.interpolated_pts_with_end_caps_np_arr
            structure_trimesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                                    interp_intra_slice_dist,
                                    structure_threeDdata_array_fully_interpolated_with_end_caps,
                                    radius_for_normals_estimation,
                                    max_nn_for_normals_estimation
                                    )
            structure_trimesh_prostate_frame = translate_mesh(structure_trimesh, -prostate_centroid)
        else: 
            pass
        
        if selected_structure_ref_type == rectum_ref:
            
            
            rectum_point_sup = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Structure centroid pts'][0]
            rectum_point_inf = pydicom_item[selected_structure_ref_type][selected_structure_structure_index]['Structure centroid pts'][-1]
            rectum_point_sup_z_aligned_with_rectum_point_inf = copy.deepcopy(rectum_point_sup)
            rectum_point_sup_z_aligned_with_rectum_point_inf[0] = rectum_point_inf[0]
            rectum_point_sup_z_aligned_with_rectum_point_inf[1] = rectum_point_inf[1]
            rectum_point_sup_z_aligned = rectum_point_sup_z_aligned_with_rectum_point_inf
            rectum_point_sup_z_aligned_prostate_frame = rectum_point_sup_z_aligned - prostate_centroid
            rectum_point_inf_prostate_frame = rectum_point_inf - prostate_centroid

            # Set the two othe points that will define the TRUS sagittal plane
            if use_natural_TRUS_origin_for_transducer_sagittal_plane == False:
                transducer_saggital_plane_point_prostate_frame_inf = rectum_point_inf_prostate_frame
                transducer_saggital_plane_point_prostate_frame_sup = rectum_point_sup_z_aligned_prostate_frame

            elif use_natural_TRUS_origin_for_transducer_sagittal_plane == True:
                transducer_saggital_plane_point_prostate_frame_inf = np.array([0,0, rectum_point_inf[2]]) - prostate_centroid
                transducer_saggital_plane_point_prostate_frame_sup = np.array([0,0, rectum_point_sup[2]]) - prostate_centroid
            
            transducer_saggital_plane_point_prostate_frame_inf_pcd = point_containment_tools.create_point_cloud(transducer_saggital_plane_point_prostate_frame_inf.reshape(1, -1), color = np.array([1,0,0]))
            transducer_saggital_plane_point_prostate_frame_sup_pcd = point_containment_tools.create_point_cloud(transducer_saggital_plane_point_prostate_frame_sup.reshape(1, -1), color = np.array([1,0,0]))

            non_dil_list_of_pcds.append(transducer_saggital_plane_point_prostate_frame_inf_pcd)
            non_dil_list_of_pcds.append(transducer_saggital_plane_point_prostate_frame_sup_pcd)

            min_z_rectum = np.min(structure_inter_slice_interp_np_arr_prostate_coords[:, 2])
        
        structure_trimesh_objs_list_of_dicts.append({'Structure ID': selected_structure_ID, 
                                                     'Structure trimesh': structure_trimesh_prostate_frame, 
                                                     'Color': pcd_color_arr, 
                                                     'Struct type': selected_structure_ref_type})

            


    # Assuming prostate_grid_template_lattice_dataframe, prostate_centroid, and prostate_points are already defined:
    coord_1_5_pos_in_prostate_coords = max_y_prostate + y_shift_for_1_5_coord_from_prostate_max_post
    grid_position_D_1_5_in_prostate_coords = np.array([0,coord_1_5_pos_in_prostate_coords,min_z_rectum])
    
    # Translate the lattice dataframe based on specified labels
    prostate_grid_template_lattice_XYZ_aligned_dataframe = translate_lattice_based_on_labels(prostate_grid_template_lattice_dataframe, 
                                                                                            grid_position_D_1_5_in_prostate_coords, 
                                                                                            letter_label=label1, 
                                                                                            numerical_label=label2)
    # Further translate the lattice in Z
    #prostate_grid_template_lattice_XYZ_aligned_dataframe = translate_lattice_in_z(prostate_grid_template_lattice_XY_aligned_dataframe, min_z_prostate)

    prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr = prostate_grid_template_lattice_XYZ_aligned_dataframe[['X','Y','Z']].to_numpy()
    #prostate_grid_template_translation_distance = prostate_inter_slice_interp_np_arr_prostate_coords[:,2].min()
    #prostate_grid_template_lattice_prostate_coord_frame_arr = prostate_grid_template_lattice_arr + np.array([0,0,prostate_grid_template_translation_distance])
    
    #pcd
    prostate_grid_template_lattice_prostate_coord_frame_pcd = point_containment_tools.create_point_cloud(prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr, color = np.array([0,1,0]))
    non_dil_list_of_pcds.append(prostate_grid_template_lattice_prostate_coord_frame_pcd)

    trus_plane_sagittal_contour_plot_list_of_dicts = []
    transverse_contour_plot_list_of_dicts = []
    dil_specific_pointclouds_list = []
    dil_pcd_color_arr = structs_referenced_dict[dil_ref]['PCD color']
    for specific_dil_index, specific_dil_structure in enumerate(pydicom_item[dil_ref]):
        sp_dil_guidance_map_max_planes_dataframe = specific_dil_structure["Biopsy optimization: guidance map max-planes dataframe"]
        optimal_positions_df = specific_dil_structure['Biopsy optimization: Optimal biopsy location dataframe']
        dil_id_from_pydicom = specific_dil_structure['ROI']
        # Extract the base point from optimal_positions_df using the first matched index
        #sp_dil_optimal_positions_df = optimal_positions_df[optimal_positions_df['Relative DIL index'] == specific_dil_index]
        sp_dil_optimal_coordinate = np.array([
            optimal_positions_df.at[0, 'Test location (Prostate centroid origin) (X)'],
            optimal_positions_df.at[0, 'Test location (Prostate centroid origin) (Y)'],
            optimal_positions_df.at[0, 'Test location (Prostate centroid origin) (Z)']
        ])
        sp_dil_id = optimal_positions_df.at[0, 'Relative DIL ID']

        dil_inter_slice_interp_np_arr = specific_dil_structure['Inter-slice interpolation information'].interpolated_pts_np_arr
        dil_inter_slice_interp_np_arr_prostate_coords = dil_inter_slice_interp_np_arr - prostate_centroid
        
        #pcd 
        dil_inter_slice_interp_prostate_coords_pcd = point_containment_tools.create_point_cloud(dil_inter_slice_interp_np_arr_prostate_coords, color = np.array([0,0,1]))
        optimal_pos_pcd = point_containment_tools.create_point_cloud(sp_dil_optimal_coordinate.reshape(1, -1), color = np.array([0,1,0]))
        dil_specific_pointclouds_list.append(dil_inter_slice_interp_prostate_coords_pcd)
        dil_specific_pointclouds_list.append(optimal_pos_pcd)


        transducer_plane_df, transducer_plane_normal, euler_angles, euler_convention_str = generate_grid_dataframe(transducer_saggital_plane_point_prostate_frame_sup, 
                                                                                             transducer_saggital_plane_point_prostate_frame_inf, 
                                                                                             transducer_plane_grid_spacing, 
                                                                                             entire_lattice_df, 
                                                                                             sp_dil_optimal_coordinate)
        transformed_grid_df, transformed_optimal_point, rotation_matrix = transform_grid_and_point_for_plotting(transducer_plane_df, transducer_plane_normal, sp_dil_optimal_coordinate)

        # pcd 
        #transducer_plane_pcd = point_containment_tools.create_point_cloud(transducer_plane_df[['Transducer plane point (X)', 'Transducer plane point (Y)', 'Transducer plane point (Z)']].to_numpy(), color = np.array([0,0,1]))
        #transducer_plane_rotated_pcd = point_containment_tools.create_point_cloud(transformed_grid_df[['Transformed X', 'Transformed Y', 'Transformed Z']].to_numpy(), color = np.array([0,0,1]))
        #transformed_optimal_point_pcd = point_containment_tools.create_point_cloud(transformed_optimal_point.reshape(1, -1), color = np.array([1,0,0]))
        transducer_plane_pcd_colored = dataframe_to_point_cloud(transducer_plane_df, 
                                                                'Transducer plane point (X)', 
                                                                'Transducer plane point (Y)', 
                                                                'Transducer plane point (Z)')
        dil_specific_pointclouds_list.append(transducer_plane_pcd_colored)

        # Using the function to find the nearest neighbors
        prostate_template_points = prostate_grid_template_lattice_XYZ_aligned_dataframe[['X', 'Y', 'Z']].values
        nearest_indices = find_nearest_neighbors_sklearn(prostate_template_points, sp_dil_optimal_coordinate, k=1)
        nearest_template_points = prostate_template_points[nearest_indices]


        # Project, transform, and annotate lines
        nearest_prostate_template_lines_contour_plot_coords_list_of_dicts, nearest_prostate_template_lines_transducer_plane_projected_list_of_dicts, nearest_prostate_template_lines_list_of_dicts = project_and_transform_lines(nearest_template_points, 
                                            transducer_plane_normal, 
                                            sp_dil_optimal_coordinate, 
                                            rotation_matrix, 
                                            prostate_grid_template_lattice_XYZ_aligned_dataframe.iloc[nearest_indices])
        optimal_template_hole_label = "-"
        if len(nearest_prostate_template_lines_list_of_dicts) > 0:
            optimal_template_hole_label = str(nearest_prostate_template_lines_list_of_dicts[0].get("label", "-"))


        line_pts_list = [np.array([item['start'],item['end']]) for item in nearest_prostate_template_lines_list_of_dicts]
        nearest_template_lines_colors_list = [np.array([1,0,1]) for i in range(len(line_pts_list))]
        #nearest_template_lines_lineset = create_lines_for_open3d(line_pts_list, nearest_template_lines_colors_list)
        nearest_template_lines_cylinder_mesh_list = create_thick_lines_as_cylinders(line_pts_list, nearest_template_lines_colors_list, radius=0.25)
        #nearest_template_lines_cylinder_mesh_flat_list = flatten_geometry_list(nearest_template_lines_cylinder_mesh_list)
        dil_specific_pointclouds_list.extend(nearest_template_lines_cylinder_mesh_list)

        # dil trimesh
        #dil_trimesh = specific_dil_structure['Structure OPEN3D triangle mesh object']
        dil_interpolation_information = specific_dil_structure["Intra-slice interpolation information"]
        dil_threeDdata_array_fully_interpolated_with_end_caps = dil_interpolation_information.interpolated_pts_with_end_caps_np_arr
        dil_trimesh, _ = misc_tools.compute_structure_triangle_mesh(interp_inter_slice_dist, 
                                interp_intra_slice_dist,
                                dil_threeDdata_array_fully_interpolated_with_end_caps,
                                radius_for_normals_estimation,
                                max_nn_for_normals_estimation
                                )
        dil_trimesh_prostate_frame = translate_mesh(dil_trimesh, -prostate_centroid)

        if plot_open3d_structure_set_complete_demonstration_bool == True:
            prostate_frame_representation_pcd_list = non_dil_list_of_pcds + dil_specific_pointclouds_list
            plotting_funcs.plot_geometries(*prostate_frame_representation_pcd_list)

        

        # create plot
        contour_plot = plot_transformed_contour(transformed_grid_df)

        contour_plot.update_layout(
                title=dict(
                    text=f"TRUS plane guidance map - {patientUID} - {sp_dil_id}"
                )
            )


        # prostate_mesh_slice_pts = slice_mesh_fast(prostate_trimesh_prostate_frame, transducer_plane_normal, sp_dil_optimal_coordinate)
        # prostate_mesh_slice_pts_transformed = transform_points(prostate_mesh_slice_pts, rotation_matrix, transducer_plane_df)
        # prostate_mesh_slice_pts_transformed_contour_plot_coords = prostate_mesh_slice_pts_transformed[:,[2,1]]
        # #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
        # contour_plot = plot_transformed_mesh_slice(contour_plot, prostate_mesh_slice_pts_transformed_contour_plot_coords, 'Prostate', color_input = 'red')
        
        rectum_plus_prostate_pts_contour_plot_coords_list = []
        for trimesh_dict in structure_trimesh_objs_list_of_dicts:
            contour_color = trimesh_dict['Color']
            structure_trimesh = trimesh_dict['Structure trimesh']
            structure_ID = trimesh_dict['Structure ID']
            struct_type = trimesh_dict['Struct type']

            structure_mesh_slice_pts = slice_mesh_fast_v2(structure_trimesh, transducer_plane_normal, sp_dil_optimal_coordinate)
            if len(structure_mesh_slice_pts) != 0:
                structure_mesh_slice_pts_transformed = transform_points(structure_mesh_slice_pts, rotation_matrix, transducer_plane_df)
                structure_mesh_slice_pts_transformed_contour_plot_coords = structure_mesh_slice_pts_transformed[:,[2,1]]
                #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
                contour_plot = plot_transformed_mesh_slice(contour_plot, 
                                                       structure_mesh_slice_pts_transformed_contour_plot_coords, 
                                                       structure_ID, 
                                                       color_input = misc_tools.unnormalize_color_values(contour_color))
            
                if struct_type == oar_ref or struct_type == rectum_ref:
                    rectum_plus_prostate_pts_contour_plot_coords_list.append(structure_mesh_slice_pts_transformed_contour_plot_coords)
                if struct_type == oar_ref:
                    prostate_mesh_slice_pts_transformed_contour_plot_coords = structure_mesh_slice_pts_transformed_contour_plot_coords
            else: # no points contained in slice!
                pass
        
        dil_mesh_slice_pts = slice_mesh_fast_v2(dil_trimesh_prostate_frame, transducer_plane_normal, sp_dil_optimal_coordinate)
        if len(dil_mesh_slice_pts) != 0:
            dil_mesh_slice_pts_transformed = transform_points(dil_mesh_slice_pts, rotation_matrix, transducer_plane_df)
            dil_mesh_slice_pts_transformed_contour_plot_coords = dil_mesh_slice_pts_transformed[:,[2,1]]
            #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
            contour_plot = plot_transformed_mesh_slice(contour_plot, dil_mesh_slice_pts_transformed_contour_plot_coords, sp_dil_id, color_input = misc_tools.unnormalize_color_values(dil_pcd_color_arr))

        

        #contour_plot = reflect_plot_about_x_axis(contour_plot) 

        #prostate_mesh_slice_pts_transformed_contour_plot_coords_reflected = copy.deepcopy(prostate_mesh_slice_pts_transformed_contour_plot_coords)
        #prostate_mesh_slice_pts_transformed_contour_plot_coords_reflected[:,0] = -prostate_mesh_slice_pts_transformed_contour_plot_coords_reflected[:,0]

        rectum_plus_prostate_pts_contour_plot_coords = np.vstack(rectum_plus_prostate_pts_contour_plot_coords_list)
        #rectum_plus_prostate_pts_contour_plot_coords_reflected = copy.deepcopy(rectum_plus_prostate_pts_contour_plot_coords)
        #rectum_plus_prostate_pts_contour_plot_coords_reflected[:,0] = -rectum_plus_prostate_pts_contour_plot_coords_reflected[:,0]


        #contour_plot = reverse_plot_axes(contour_plot, reverse_x=True, reverse_y=True)

        contour_plot = adjust_plot_area_and_reverse_axes(contour_plot, rectum_plus_prostate_pts_contour_plot_coords, margin=5, reverse_x=True, reverse_y=True)
        _show_all_axis_lines(contour_plot)

        transformed_optimal_point_contour_coord_sys = transformed_optimal_point[[2,1]]
        transformed_optimal_point_xprime = float(transformed_optimal_point[0])
        optimal_row_sagittal = {
            "label": f"Optimal ({sp_dil_id})",
            "optimal_hole": optimal_template_hole_label,
            "xprime": transformed_optimal_point_xprime,
            "zprime": transformed_optimal_point_contour_coord_sys[0],
            "yprime": transformed_optimal_point_contour_coord_sys[1]
        }
        if fire_annotation_style == "hockey":
            contour_plot = add_points_to_plot_v2(contour_plot,
                                         transformed_optimal_point_contour_coord_sys,
                                         f"Optimal ({sp_dil_id}) | [{transformed_optimal_point_contour_coord_sys[0]:.2f}, {transformed_optimal_point_contour_coord_sys[1]:.2f}] mm",
                                         legend_name = f"Optimal ({sp_dil_id})",
                                         size = 12,
                                         annotation_direction = "down",
                                         text_box_bg_color="rgba(255, 255, 255, 1)",
                                         text_box_border_color="rgba(0, 0, 0, 1)",
                                         text_color="black")
        elif fire_annotation_style == "compact_table":
            contour_plot = add_points_to_plot(contour_plot,
                                     transformed_optimal_point_contour_coord_sys,
                                     legend_name = f"Optimal ({sp_dil_id})",
                                     color = "orange",
                                     size = 12)

        fire_rows = []
        custom_height = 0
        custom_height_step = 4
        for index,penetration_depth in enumerate(biopsy_fire_travel_distances):
            custom_height = custom_height + custom_height_step*(index+1)
            needle_tip_position_before_firing = copy.deepcopy(transformed_optimal_point_contour_coord_sys)
            needle_tip_position_before_firing[0] = needle_tip_position_before_firing[0] - penetration_depth + (biopsy_needle_tip_length + biopsy_needle_compartment_length/2)

            # Compute deflection distance from optimal template hole trajectory (pink line)
            def _point_to_segment_distance(pt, start, end):
                """Euclidean distance from point to line segment in 2D."""
                seg = end - start
                seg_len_sq = np.dot(seg, seg)
                if seg_len_sq == 0:
                    return float(np.linalg.norm(pt - start)), start
                t = np.dot(pt - start, seg) / seg_len_sq
                t_clamped = np.clip(t, 0.0, 1.0)
                projection = start + t_clamped * seg
                return float(np.linalg.norm(pt - projection)), projection

            deflection_dist = float("nan")
            projection_point = None
            if len(nearest_prostate_template_lines_contour_plot_coords_list_of_dicts) > 0:
                line_coords = nearest_prostate_template_lines_contour_plot_coords_list_of_dicts[0]
                # Convert to contour frame (Z', Y')
                line_start = np.array([line_coords['start'][2], line_coords['start'][1]])
                line_end = np.array([line_coords['end'][2], line_coords['end'][1]])
                deflection_dist, projection_point = _point_to_segment_distance(needle_tip_position_before_firing, line_start, line_end)

            # Visual deflection line (projection to template line)
            if projection_point is not None:
                double_arrow_deflection_line_offset = 1  # mm offset to avoid overlap with needle tip line and align annotation
                p0 = np.array([needle_tip_position_before_firing[0], needle_tip_position_before_firing[1]])
                p1 = np.array([projection_point[0], projection_point[1]])
                # reuse generic distance arrow helper in segment mode
                contour_plot = add_distance_annotation(contour_plot,
                                                       np.vstack([p0, p1]),
                                                       start_point=p0,
                                                       end_point=p1,
                                                       segment_offset=(double_arrow_deflection_line_offset, 0.0),
                                                       arrow_color='black',
                                                       line_width=3,
                                                       line_dash='solid',
                                                       show_text=False,
                                                       show_legend=True,
                                                       legend_name="Deflection line")

            # Explicit axis/frame labeling (transducer-plane contour frame; origin at prostate centroid)
            prostate_apex_zprime = float(np.min(prostate_mesh_slice_pts_transformed_contour_plot_coords[:,0]))
            depth_from_apex = needle_tip_position_before_firing[0] - prostate_apex_zprime
            fire_label = (
                f"Tip before fire | Penetration depth {penetration_depth:.1f} mm<br>"
                f"[Z'={needle_tip_position_before_firing[0]:.2f} mm, "
                f"Y'={needle_tip_position_before_firing[1]:.2f} mm] (transducer-plane frame)<br>"
                f"Deflection from template line: {deflection_dist:.2f} mm<br>"
                f"Tip depth from apex: {depth_from_apex:.2f} mm"
            )
            legend_label = f"Fire pos. ({penetration_depth:.1f} mm)"
            line_offset = 1  # mm offset for the line to avoid overlap with the point
            fire_rows.append({
                "penetration_depth": penetration_depth,
                "optimal_hole": optimal_template_hole_label,
                "xprime": transformed_optimal_point_xprime,
                "zprime": needle_tip_position_before_firing[0],
                "yprime": needle_tip_position_before_firing[1],
                "deflection": deflection_dist,
                "depth_from_apex": depth_from_apex
            })

            if fire_annotation_style == "hockey":
                contour_plot = add_points_to_plot_v2(contour_plot, 
                                                    needle_tip_position_before_firing, 
                                                    fire_label, 
                                                    color = 'red', 
                                                    symbol = 'arrow-bar-left', 
                                                    custom_height = custom_height, 
                                                    legend_name = legend_label,
                                                    color_index = index,
                                                    annotation_x_offset = line_offset,
                                                    text_box_bg_color="rgba(255, 255, 255, 1)",
                                                    text_box_border_color="rgba(0, 0, 0, 1)",
                                                    text_color="black")
            elif fire_annotation_style == "compact_table":
                marker_colors = ['#333333', '#FF7F50', '#008080']
                contour_plot.add_trace(go.Scatter(
                    x=[needle_tip_position_before_firing[0]],
                    y=[needle_tip_position_before_firing[1]],
                    mode='markers',
                    marker=dict(color=marker_colors[index % len(marker_colors)], size=10, symbol='arrow-bar-left'),
                    name=legend_label
                ))

        contour_plot = add_lines_to_contour_plot(contour_plot, nearest_prostate_template_lines_contour_plot_coords_list_of_dicts)


        contour_plot = add_x_bounds_with_annotations(contour_plot, 
                                        prostate_mesh_slice_pts_transformed_contour_plot_coords, 
                                        y_position = 1.0,
                                        x_offset = -1.5,  
                                        label_yanchor = "top",
                                        max_x_label="Base", 
                                        min_x_label="Apex", 
                                        line_color_max='black', 
                                        line_color_min='black', 
                                        line_style_max='dot',
                                        line_style_min='dot', 
                                        line_width=3 
                                        )


        contour_plot = add_distance_annotation(contour_plot, 
                                               prostate_mesh_slice_pts_transformed_contour_plot_coords, 
                                               y_position=-0.075, 
                                               arrow_color='black',
                                               x_offset=0,
                                               text_box_bg_color="rgba(255, 255, 255, 1)",
                                               text_box_border_color="rgba(0, 0, 0, 1)",
                                               text_color="black")

        if simple_angle_display_option_bool == False:
            contour_plot = add_euler_angles_to_plot_v3(contour_plot, 
                                                    euler_angles,
                                                    euler_convention_str, 
                                                    position='bottom right')
        elif simple_angle_display_option_bool == True:
            contour_plot = add_sagittal_angle_to_plot(contour_plot, 
                                                                 z_angle, 
                                                                 position='bottom right')

        
        z_angle = euler_angles[2]  # Assuming euler_angles is accessible and index 2 is Z

        _anchor_colorbars_bottom_right(contour_plot)

        contour_plot = set_square_aspect_ratio(contour_plot)
        

        if fire_annotation_style == "compact_table":
            contour_plot = add_compact_fire_positions_table(contour_plot,
                                                            fire_rows,
                                                            position=fire_table_position,
                                                            frame_label="Transducer plane frame (Z', Y')",
                                                            optimal_row=optimal_row_sagittal)

        sp_dil_contour_plot_dict = {"DIL ID": dil_id_from_pydicom,
                                    "Contour plot": contour_plot}
        trus_plane_sagittal_contour_plot_list_of_dicts.append(sp_dil_contour_plot_dict)







        ###

        ### TRANSVERSE ###

        ###







        plane_specific_guidance_map_max_planes_dataframe = sp_dil_guidance_map_max_planes_dataframe[sp_dil_guidance_map_max_planes_dataframe['Patient plane'].str.contains('Transverse')]

        #plane_specific_guidance_map_max_planes_dataframe = sp_dil_guidance_map_max_planes_dataframe[sp_dil_guidance_map_max_planes_dataframe['Patient plane'] == plane]
        #hor_axis_column_name = plane_specific_guidance_map_max_planes_dataframe.sample().reset_index(drop=True).at[0,'Coord 1 name']
        #vert_axis_column_name = plane_specific_guidance_map_max_planes_dataframe.sample().reset_index(drop=True).at[0,'Coord 2 name']
        #const_axis_column_name = plane_specific_guidance_map_max_planes_dataframe.sample().reset_index(drop=True).at[0,'Const coord name']
        
        contour_plot_transverse = go.Figure()

        sp_dil_optimal_coordinate_transverse_contour_plot_coords = sp_dil_optimal_coordinate[[0,1]]
        optimal_row_transverse = {
            "label": f"Optimal ({sp_dil_id})",
            "optimal_hole": optimal_template_hole_label,
            "xprime": transformed_optimal_point_xprime,
            "zprime": transformed_optimal_point_contour_coord_sys[0],
            "yprime": transformed_optimal_point_contour_coord_sys[1]
        }
        if fire_annotation_style == "hockey":
            contour_plot_transverse = add_points_to_plot_v2(contour_plot_transverse,
                                         sp_dil_optimal_coordinate_transverse_contour_plot_coords,
                                         f"Optimal ({sp_dil_id}) | [{sp_dil_optimal_coordinate_transverse_contour_plot_coords[0]:.2f}, {sp_dil_optimal_coordinate_transverse_contour_plot_coords[1]:.2f}] mm",
                                         legend_name = f"Optimal ({sp_dil_id})",
                                         size = 12)
        elif fire_annotation_style == "compact_table":
            contour_plot_transverse = add_points_to_plot(contour_plot_transverse,
                                         sp_dil_optimal_coordinate_transverse_contour_plot_coords,
                                         legend_name = f"Optimal ({sp_dil_id})",
                                         color = "orange",
                                         size = 12)

        origin = np.array([0,0]) 
        contour_plot_transverse = add_points_to_plot(contour_plot_transverse, 
                                     origin, 
                                     legend_name = f"Prostate centroid projection",
                                     color = "black",
                                    size = 10)

        transverse_plane_normal = np.array([0.,0.,1.])
        alternate_prostate_slice_shift_from_prostate_apex = 10 # ie. 1 cm from apex
        rectum_plus_prostate_mesh_transverse_slice_pts_prostate_coord_frame_list = []
        for trimesh_dict in structure_trimesh_objs_list_of_dicts:
            contour_color = trimesh_dict['Color']
            structure_trimesh = trimesh_dict['Structure trimesh']
            structure_ID = trimesh_dict['Structure ID']
            struct_type = trimesh_dict['Struct type']

            structure_mesh_slice_pts = slice_mesh_fast_v2(structure_trimesh, transverse_plane_normal, sp_dil_optimal_coordinate)
            if len(structure_mesh_slice_pts) != 0:

                #structure_mesh_slice_pts_transformed = transform_points(structure_mesh_slice_pts, rotation_matrix, transducer_plane_df)
                structure_mesh_slice_pts_transverse_contour_plot_coords = structure_mesh_slice_pts[:,[0,1]]
                #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
                contour_plot_transverse = plot_transformed_mesh_slice(contour_plot_transverse, 
                                                       structure_mesh_slice_pts_transverse_contour_plot_coords, 
                                                       structure_ID, 
                                                       color_input = misc_tools.unnormalize_color_values(contour_color))
            if struct_type == oar_ref:
                contour_color_alternate = np.array([0,0,0])
                alternate_z_slice_prostate_location = min_z_prostate + alternate_prostate_slice_shift_from_prostate_apex
                structure_mesh_slice_pts = slice_mesh_fast_v2(structure_trimesh, transverse_plane_normal, alternate_z_slice_prostate_location)
                if len(structure_mesh_slice_pts) != 0:
                    #structure_mesh_slice_pts_transformed = transform_points(structure_mesh_slice_pts, rotation_matrix, transducer_plane_df)
                    structure_mesh_slice_pts_transverse_contour_plot_coords = structure_mesh_slice_pts[:,[0,1]]
                    #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
                    contour_plot_transverse = plot_transformed_mesh_slice(contour_plot_transverse, 
                                                            structure_mesh_slice_pts_transverse_contour_plot_coords, 
                                                            f'{structure_ID} {alternate_prostate_slice_shift_from_prostate_apex:.1f} mm shift from apex', 
                                                            color_input = misc_tools.unnormalize_color_values(contour_color_alternate))
            if struct_type == oar_ref or struct_type == rectum_ref:
                rectum_plus_prostate_mesh_transverse_slice_pts_prostate_coord_frame_list.append(structure_mesh_slice_pts_transverse_contour_plot_coords)

        rectum_plus_prostate_mesh_transverse_slice_pts_prostate_coord_frame = np.vstack(rectum_plus_prostate_mesh_transverse_slice_pts_prostate_coord_frame_list)


        dil_mesh_slice_pts = slice_mesh_fast_v2(dil_trimesh_prostate_frame, transverse_plane_normal, sp_dil_optimal_coordinate)
        if len(dil_mesh_slice_pts) != 0:
            #dil_mesh_slice_pts_transformed = transform_points(dil_mesh_slice_pts, rotation_matrix, transducer_plane_df)
            dil_mesh_slice_pts_transverse_contour_plot_coords = dil_mesh_slice_pts[:,[0,1]]
            #dil_mesh_slice_pts_transformed_contour_plot_coords = dil_mesh_slice_pts_transformed[:,[2,1]]
            #mesh_slice_pts_transformed_ordered = solve_tsp_google(mesh_slice_pts_transformed)
            contour_plot_transverse = plot_transformed_mesh_slice(contour_plot_transverse, 
                                                                  dil_mesh_slice_pts_transverse_contour_plot_coords, 
                                                                  sp_dil_id, 
                                                                  color_input = misc_tools.unnormalize_color_values(dil_pcd_color_arr))
        
        ## this was code to plot the nearest z slice of the dil structure instead of slicing the trimesh objects using slice_mesh_fast
        # interslice_interpolation_information = specific_dil_structure["Inter-slice interpolation information"]                        
        # interpolated_pts_np_arr = interslice_interpolation_information.interpolated_pts_np_arr
        # interpolated_zvals_list = interslice_interpolation_information.zslice_vals_after_interpolation_list
        # intraslice_interpolated_zslices_list = dil_interpolation_information.interpolated_pts_list
        # sp_dil_optimal_coordinate_org_frame = sp_dil_optimal_coordinate + prostate_centroid
        # closest_z_index, closest_z_val = point_containment_tools.take_closest_numpy(interpolated_zvals_list, [sp_dil_optimal_coordinate_org_frame[2]])
        # transverse_slice_of_dil_at_optimal_depth = np.array(intraslice_interpolated_zslices_list[closest_z_index[0]])
        # transverse_slice_of_dil_at_optimal_depth_in_prostate_frame = transverse_slice_of_dil_at_optimal_depth - prostate_centroid
        # transverse_slice_of_dil_at_optimal_depth_in_prostate_frame_with_first_point_appended_for_connection = np.append(transverse_slice_of_dil_at_optimal_depth_in_prostate_frame, transverse_slice_of_dil_at_optimal_depth_in_prostate_frame[[0],:], axis = 0)

        # contour_plot_transverse = plot_transformed_mesh_slice(contour_plot_transverse, 
        #                                                           transverse_slice_of_dil_at_optimal_depth_in_prostate_frame, 
        #                                                           sp_dil_id, 
        #                                                           color_input = np.array([0,255,0]))


        # Add contour plot of the values of the max plane of the optimization

        contour_plot_transverse = plot_transverse_contour(contour_plot_transverse,
                                                          plane_specific_guidance_map_max_planes_dataframe,
                                                          colorbar_title="Containment proportion",
                                                          colorbar_title_font_size=colorbar_title_font_size
                                                          )
        

        contour_plot_transverse.update_layout(
                title=dict(
                    text=f"Transverse (Max) plane - {patientUID} - {sp_dil_id}"
                )
            )
        _anchor_colorbars_bottom_right(contour_plot_transverse)


        # Example data preparation
        # prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr_XY = prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr[:, [0,1]]  # Extract only the XY coordinates
        # nearest_points = nearest_template_points[:, [0,1]]  # Assume nearest_template_points is already filtered to the relevant coordinates
        # labels = prostate_grid_template_lattice_XYZ_aligned_dataframe.loc[nearest_indices].apply(lambda x: f"{x['Label 1']}-{x['Label 2']}", axis=1).tolist()

        # Assuming 'contour_plot_transverse' is already initialized and is a Plotly figure object
        # contour_plot_transverse = add_transverse_contour_plot_elements(contour_plot_transverse, 
        #                                                                prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr_XY, 
        #                                                                nearest_points, 
        #                                                                labels)
        
        contour_plot_transverse = add_perineal_template_lattice_to_transverse_contour_plot_outer_annotations_only(contour_plot_transverse, 
                                                                                           prostate_grid_template_lattice_XYZ_aligned_dataframe, 
                                                                                           nearest_template_points,
                                                                                           template_label_font_size=template_label_font_size)

        

        

        z_angle = euler_angles[2]  # Assuming euler_angles is accessible and index 2 is Z
        contour_plot_transverse = add_z_angle_line_to_plot(contour_plot_transverse, transducer_saggital_plane_point_prostate_frame_sup, z_angle)

        if simple_angle_display_option_bool == False:
            contour_plot_transverse = add_euler_angles_to_plot_v3(contour_plot_transverse, 
                                                    euler_angles,
                                                    euler_convention_str, 
                                                    position='bottom right')
        elif simple_angle_display_option_bool == True:
            contour_plot_transverse = add_sagittal_angle_to_plot(contour_plot_transverse, 
                                                                 z_angle, 
                                                                 position='bottom right')

        prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr_XY = prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr[:, [0,1]]  # Extract only the XY coordinates
        rectum_plus_prostate_plus_perineal_template_pts_prostate_frame_coords = np.vstack([rectum_plus_prostate_mesh_transverse_slice_pts_prostate_coord_frame,prostate_grid_template_lattice_XYZ_aligned_prostate_coord_frame_arr_XY])
        bounds_points_xy = rectum_plus_prostate_plus_perineal_template_pts_prostate_frame_coords[:, :2]

        contour_plot_transverse = adjust_plot_area_and_reverse_axes(contour_plot_transverse, 
                                                                    bounds_points_xy, 
                                                                    margin=5, 
                                                                    reverse_x = False, 
                                                                    reverse_y = True)
        _show_all_axis_lines(contour_plot_transverse)

        if draw_orientation_diagram:
            contour_plot_transverse = add_angle_orientation_diagram(contour_plot_transverse, position = (0.8,0.05))

        contour_plot_transverse = set_square_aspect_ratio(contour_plot_transverse)

        

        if fire_annotation_style == "compact_table":
            contour_plot_transverse = add_compact_fire_positions_table(contour_plot_transverse,
                                                                       fire_rows,
                                                                       position=fire_table_position,
                                                                       frame_label="Transducer plane frame (Z', Y')",
                                                                       optimal_row=optimal_row_transverse)

        sp_dil_transverse_contour_plot_dict = {"DIL ID": dil_id_from_pydicom,
                                    "Contour plot": contour_plot_transverse}
        transverse_contour_plot_list_of_dicts.append(sp_dil_transverse_contour_plot_dict)
        


    return trus_plane_sagittal_contour_plot_list_of_dicts, transverse_contour_plot_list_of_dicts
