import numpy as np
import copy

def calculate_gradient_lattices(dose_values_3d, pixel_spacing, grid_frame_offset_vec_list):
    """
    Calculate the gradient lattice and gradient norm lattice for a 3D dose map.

    Parameters:
    - phys_space_dose_map_3d_arr (numpy.ndarray): 3D array containing physical space and dose values. Each entry has physical coordinates (x, y, z) and a dose value.
    - pixel_spacing (tuple): (dy, dx) spacing values derived from DICOM pixel spacing.
    - grid_frame_offset_vec_list (list or numpy.ndarray): List or array of z-coordinates derived from the Grid Frame Offset Vector (DICOM tag 3004,000C).

    Returns:
    - gradient_vector_lattice (numpy.ndarray): 4D array (Nx, Ny, Nz, 3) of gradient vectors at each point.
    - gradient_norm_lattice (numpy.ndarray): 3D array (Nx, Ny, Nz) of gradient norms at each point.
    """

    # Spacing from inputs
    dx = pixel_spacing[1]  # Column spacing (x-axis)
    dy = pixel_spacing[0]  # Row spacing (y-axis)
    dz = np.array(grid_frame_offset_vec_list)  # Z-coordinates

    # Ensure z spacing matches size of z dimension
    if len(dz) != dose_values_3d.shape[0]:
        raise ValueError("Length of grid_frame_offset_vec_list must match the z-dimension of the dose grid.")

    # Calculate gradients using physical spacings
    grad_x, grad_y, grad_z = np.gradient(dose_values_3d, dz, dx, dy)

    # Combine gradients into a single array (Nx, Ny, Nz, 3)
    gradient_vector_lattice = np.stack((grad_x, grad_y, grad_z), axis=-1)

    # Compute the norm of the gradient at each point (Nx, Ny, Nz)
    gradient_norm_lattice = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Avoid division by zero by setting zero norms to 1 temporarily (will result in zero normalized vectors)
    gradient_norm_safe = np.where(gradient_norm_lattice == 0, 1, gradient_norm_lattice)

    # Normalize the gradient vectors
    normalized_gradient_vector_lattice = gradient_vector_lattice / gradient_norm_safe[..., None]

    return gradient_vector_lattice, gradient_norm_lattice, normalized_gradient_vector_lattice




def map_pixel_to_physical(conversion_matrix, grid_shape):
    """Map pixel indices to physical space for a single slice."""
    rows, cols = grid_shape
    row_indices, col_indices = np.meshgrid(range(rows), range(cols), indexing='ij')
    pixel_coords = np.stack([col_indices.flatten(), row_indices.flatten(), np.zeros(rows * cols), np.ones(rows * cols)])
    physical_coords = np.dot(conversion_matrix, pixel_coords)
    # Ensure correct pairing: row index (j), column index (i), x, y, z
    return np.hstack([
        row_indices.flatten()[:, None],  # Row index (j)
        col_indices.flatten()[:, None],  # Column index (i)
        physical_coords[:3].T            # x, y, z coordinates
    ])


def process_single_slice(slice_index, dose_slice, scaling_factor, base_physical_coords, z_offset):
    """Process a single slice: apply scaling and add z-offset."""
    # Flatten dose values and scale
    dose_flattened = dose_slice.flatten(order='C') * scaling_factor

    # Create a copy of the base physical coordinates to ensure isolation
    physical_coords = copy.deepcopy(base_physical_coords)
    physical_coords[:, 4] += z_offset  # Apply the Z offset for the current slice

    # Combine slice index, physical coordinates, and dose
    slice_data = np.hstack([
        np.full((physical_coords.shape[0], 1), slice_index),  # Slice index
        physical_coords,                                      # x, y, z
        dose_flattened[:, None]                               # Dose
    ])
    return slice_data


def build_dose_grid(dose_pixel_slices, scaling_factor, conversion_matrix, grid_frame_offset_vec_list):
    """Build the complete dose grid for a patient."""
    rows, cols = dose_pixel_slices[0].shape
    base_physical_coords = map_pixel_to_physical(conversion_matrix, (rows, cols))
    dose_grid = []

    for slice_index, dose_slice in enumerate(dose_pixel_slices):
        # Extract the Z offset for the current slice
        z_offset = grid_frame_offset_vec_list[slice_index]

        # Process the slice with its specific Z offset
        slice_data = process_single_slice(
            slice_index, dose_slice, scaling_factor, base_physical_coords, z_offset
        )
        dose_grid.append(slice_data)

    # Stack slices into a single array
    return np.array(dose_grid)










def map_gradient_to_physical_space(phys_space_dose_map_3d_arr, gradient_vector_lattice, gradient_norm_lattice, normalized_gradient_vector_lattice):
    """
    Map gradient values to physical space using the indices in phys_space_dose_map_3d_arr.

    Parameters:
        phys_space_dose_map_3d_arr (numpy.ndarray): Dose map with physical coordinates and voxel indices (slice-wise). 
            Shape: (num_slices, num_voxels_per_slice, 7).
            Columns:
                [0] - Slice index
                [1] - Row index (j)
                [2] - Column index (i)
                [3] - X-coordinate (physical space)
                [4] - Y-coordinate (physical space)
                [5] - Z-coordinate (physical space)
                [6] - Dose value
                
        gradient_vector_lattice (numpy.ndarray): Gradient vectors. Shape: (Nx, Ny, Nz, 3).
            Components:
                [:, :, :, 0] - Gradient in X (Gx)
                [:, :, :, 1] - Gradient in Y (Gy)
                [:, :, :, 2] - Gradient in Z (Gz)
                
        gradient_norm_lattice (numpy.ndarray): Gradient norms (|G|). Shape: (Nx, Ny, Nz).
        
        normalized_gradient_vector_lattice (numpy.ndarray): Normalized gradient vectors. Shape: (Nx, Ny, Nz, 3).
            Components:
                [:, :, :, 0] - Normalized Gradient in X (NGx)
                [:, :, :, 1] - Normalized Gradient in Y (NGy)
                [:, :, :, 2] - Normalized Gradient in Z (NGz)

    Returns:
        phys_space_gradient_map_3d_arr (numpy.ndarray): Updated slice-wise array with gradients and normalized gradients added.
            Shape: (num_slices, num_voxels_per_slice, 14).
            Columns:
                [0]  - Slice index
                [1]  - Row index (j)
                [2]  - Column index (i)
                [3]  - X-coordinate (physical space)
                [4]  - Y-coordinate (physical space)
                [5]  - Z-coordinate (physical space)
                [6]  - Dose value
                [7]  - Gradient in X (Gx)
                [8]  - Gradient in Y (Gy)
                [9]  - Gradient in Z (Gz)
                [10] - Gradient norm (|G|)
                [11] - Normalized Gradient in X (NGx)
                [12] - Normalized Gradient in Y (NGy)
                [13] - Normalized Gradient in Z (NGz)
    """
    # Flatten gradient arrays (slice-wise)
    Gx = gradient_vector_lattice[:, :, :, 0].reshape(-1, 1)
    Gy = gradient_vector_lattice[:, :, :, 1].reshape(-1, 1)
    Gz = gradient_vector_lattice[:, :, :, 2].reshape(-1, 1)
    G_norm = gradient_norm_lattice.reshape(-1, 1)
    NGx = normalized_gradient_vector_lattice[:, :, :, 0].reshape(-1, 1)
    NGy = normalized_gradient_vector_lattice[:, :, :, 1].reshape(-1, 1)
    NGz = normalized_gradient_vector_lattice[:, :, :, 2].reshape(-1, 1)

    # Flatten phys_space_dose_map_3d_arr slice-wise
    phys_space_flattened = phys_space_dose_map_3d_arr.reshape(-1, phys_space_dose_map_3d_arr.shape[-1])

    # Append gradients and normalized gradients
    gradient_map_flattened = np.hstack([
        phys_space_flattened,
        Gx, Gy, Gz, G_norm,  # Gradients and gradient norm
        NGx, NGy, NGz        # Normalized gradients
    ])

    # Reshape back to slice-wise structure
    return gradient_map_flattened.reshape(phys_space_dose_map_3d_arr.shape[0], phys_space_dose_map_3d_arr.shape[1], -1)



