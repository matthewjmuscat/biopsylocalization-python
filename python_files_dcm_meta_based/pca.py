from sklearn.decomposition import PCA
import numpy as np
import cupy as cp

def linear_fitter(data):
    if isinstance(data, cp.ndarray):
        data = cp.asnumpy(data)
    data = np.array(data).T
    pca = PCA(n_components=1)
    pca.fit(data)
    direction_vector = pca.components_
    #print(direction_vector)

    # create line
    origin = np.mean(data, axis=0)
    euclidian_distance = np.linalg.norm(data - origin, axis=1)
    extent = np.max(euclidian_distance)

    line = np.vstack((origin - direction_vector * extent,
                    origin + direction_vector * extent))

    return line



def vectorized_linear_fitter(data):
    """
    Computes the major axis (centroid line) for multiple trials in a vectorized manner.
    
    Parameters:
      data (cupy.ndarray or numpy.ndarray): Array of shape (num_trials, N, 3)
        containing N points in 3D for each trial.
        
    Returns:
      lines (cupy.ndarray): Array of shape (num_trials, 2, 3) where each trial has two endpoints:
         [origin - extent * principal_axis, origin + extent * principal_axis].
    """
    import cupy as cp  # assume using CuPy; use np for NumPy if needed
    
    # Ensure data is a cupy array
    if isinstance(data, np.ndarray):
        data = cp.array(data)
    
    # Compute the mean for each trial (shape: (num_trials, 1, 3))
    means = cp.mean(data, axis=1, keepdims=True)
    
    # Center the data for each trial (shape: (num_trials, N, 3))
    centered = data - means
    
    # Number of points per trial
    N = data.shape[1]
    
    # Compute covariance matrices for each trial: shape (num_trials, 3, 3)
    # Covariance matrix for trial i: cov_i = (X_i^T X_i)/(N-1)
    cov = cp.matmul(centered.transpose(0, 2, 1), centered) / (N - 1)
    
    # Compute eigen decomposition for each covariance matrix
    # cp.linalg.eigh can handle a stack of matrices (batched operation)
    eigenvals, eigenvecs = cp.linalg.eigh(cov)  # eigenvals: (num_trials, 3), eigenvecs: (num_trials, 3, 3)
    
    # The principal (major) axis is the eigenvector corresponding to the largest eigenvalue.
    # Since eigh returns eigenvalues in ascending order, select the last column.
    principal_axes = eigenvecs[:, :, -1]  # shape: (num_trials, 3)
    
    # Compute the extent for each trial as the maximum distance of points from the mean.
    distances = cp.linalg.norm(centered, axis=2)  # shape: (num_trials, N)
    extent = cp.max(distances, axis=1, keepdims=True)  # shape: (num_trials, 1)
    
    # Create the endpoints of the line for each trial:
    lower_endpoint = means[:, 0, :] - principal_axes * extent  # shape: (num_trials, 3)
    upper_endpoint = means[:, 0, :] + principal_axes * extent  # shape: (num_trials, 3)
    
    # Stack to form an array of shape (num_trials, 2, 3)
    lines = cp.stack([lower_endpoint, upper_endpoint], axis=1)
    return lines
