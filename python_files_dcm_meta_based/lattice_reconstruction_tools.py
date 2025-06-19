import pydicom
import numpy as np
import os

def reconstruct_mr_lattice(dicom_directory):
    """
    Reconstructs a cubic lattice of MR data from a series of DICOM slices.
    
    Parameters:
    dicom_directory (str): Directory containing the MR DICOM files.

    Returns:
    numpy.ndarray: 3D cubic lattice (voxel grid) of MR image data.
    dict: DICOM metadata containing pixel spacing and slice thickness.
    """
    dicom_files = [os.path.join(dicom_directory, f) for f in os.listdir(dicom_directory) if f.endswith('.dcm')]
    
    # Read the first DICOM file to initialize parameters
    ref_ds = pydicom.dcmread(dicom_files[0])
    
    # Get metadata for spatial information
    pixel_spacing = ref_ds.PixelSpacing  # Pixel spacing in X and Y directions
    slice_thickness = ref_ds.SliceThickness  # Z spacing between slices
    slice_positions = []  # To store slice positions along the Z axis
    slices = []  # To store each slice's pixel data
    
    # Loop through DICOM files, extract pixel data and slice position
    for dicom_file in dicom_files:
        ds = pydicom.dcmread(dicom_file)
        pixel_array = ds.pixel_array
        
        # Apply rescaling if needed
        if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
            pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        
        # Append the slice data and position
        slices.append(pixel_array)
        slice_positions.append(float(ds.ImagePositionPatient[2]))  # Z-position of the slice
    
    # Sort the slices based on their Z position
    slice_positions, slices = zip(*sorted(zip(slice_positions, slices)))
    
    # Stack slices into a 3D numpy array (cubic lattice)
    cubic_lattice = np.stack(slices, axis=-1)  # Create 3D volume by stacking along Z axis
    
    # Metadata for voxel spacing
    voxel_spacing = {
        'pixel_spacing': pixel_spacing,  # XY pixel spacing
        'slice_thickness': slice_thickness  # Z spacing
    }
    
    return cubic_lattice, voxel_spacing



### THIS FUNCTION DOES NOT SEEM TO RECONSTRUCT THE LATTICE PROPERLY!!!
def reconstruct_mr_lattice_with_coordinates_from_dict(mr_adc_ref_subdict, rescale_values = False):
    """
    Reconstructs a cubic lattice of MR data from the provided mr_adc_ref_subdict and outputs it as an (N, 4) array,
    where each row contains the (x, y, z, ADC value) for each voxel.

    Parameters:
    mr_adc_ref_subdict (dict): Dictionary containing MR ADC data and relevant DICOM metadata.

    Returns:
    numpy.ndarray: (N, 4) array, where each row is (x, y, z, ADC value).
    dict: Metadata containing pixel spacing and slice thickness.
    """
    # Extract the relevant data from the subdict
    pixel_arrays = mr_adc_ref_subdict["Pixel arr (all slices)"]  # 3D numpy array (rows, cols, slices)
    rwv_slope = mr_adc_ref_subdict["RWVSlope (all slices)"]  # 1D array of slopes for each slice
    rwv_intercept = mr_adc_ref_subdict["RWVIntercept (all slices)"]  # 1D array of intercepts for each slice
    pixel_spacing = mr_adc_ref_subdict["Pixel spacing"]  # 1D array (row_spacing, col_spacing)
    slice_thickness = mr_adc_ref_subdict["Slice thickness"]  # Scalar value for slice thickness
    image_positions = mr_adc_ref_subdict["Image position patient (all slices)"]  # 2D array (slice_position)

    # Get the number of slices and shape of each slice
    num_slices = pixel_arrays.shape[2]  # Third dimension is the number of slices
    slice_shape = pixel_arrays.shape[:2]  # First two dimensions are rows and cols (height, width)
    
    # Apply the real-world value mapping (ADC conversion) for each slice
    if rescale_values == True:
        for i in range(num_slices):
            pixel_arrays[:, :, i] = pixel_arrays[:, :, i] * rwv_slope[i] + rwv_intercept[i]

    # Generate x, y, z coordinates
    row_spacing, col_spacing = pixel_spacing  # XY pixel spacing (in mm)

    # Generate x, y, z coordinates for the whole volume
    x_coords = np.arange(slice_shape[1]) * col_spacing  # X coordinates based on columns
    y_coords = np.arange(slice_shape[0]) * row_spacing  # Y coordinates based on rows
    z_coords = image_positions[:, 2]  # Z coordinates from the ImagePositionPatient

    # Create a meshgrid for x, y, and z coordinates
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Flatten the 3D grids and the ADC values to create (N, 4) array
    voxel_positions = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), pixel_arrays.ravel()]).T


    return voxel_positions




def reconstruct_mr_lattice_with_coordinates_from_dict_v2(mr_adc_ref_subdict, filter_out_negatives = True, set_negative_to_value=None):
    """
    Reconstructs a cubic lattice of MR data from the provided mr_adc_ref_subdict and outputs it as an (N, 4) array,
    where each row contains the (x, y, z, ADC value) for each voxel.

    Parameters:
    mr_adc_ref_subdict (dict): Dictionary containing MR ADC data and relevant DICOM metadata.
    filter_out_negatives (bool): If True, removes voxels with negative ADC values.
    set_negative_to_value (float or None): If not None, sets negative ADC values to this number instead of filtering.


    Returns:
    numpy.ndarray: (N, 4) array, where each row is (x, y, z, ADC value).
    """
    # Extract the relevant data from the subdict
    pixel_arrays = mr_adc_ref_subdict["Pixel arr (all slices)"]  # 3D numpy array (rows, cols, slices)
    rwv_slope = mr_adc_ref_subdict["RWVSlope (all slices)"]  # 1D array of slopes for each slice
    rwv_intercept = mr_adc_ref_subdict["RWVIntercept (all slices)"]  # 1D array of intercepts for each slice
    pixel_spacing = mr_adc_ref_subdict["Pixel spacing"]  # 1D array (row_spacing, col_spacing)
    slice_thickness = mr_adc_ref_subdict["Slice thickness"]  # Scalar value for slice thickness
    image_positions = mr_adc_ref_subdict["Image position patient (all slices)"]  # 2D array (slice_position)
    image_orientation = mr_adc_ref_subdict["Image orientation patient"]  # 1D array of 6 values
    pixel_values_units = mr_adc_ref_subdict["Units"]  # Units of pixel array values
    
    # Get the number of slices and shape of each slice
    num_slices = pixel_arrays.shape[2]  # Third dimension is the number of slices
    slice_shape = pixel_arrays.shape[:2]  # First two dimensions are rows and cols (height, width)
    
    # Apply the real-world value mapping (ADC conversion) for each slice if not in units of mm2/s
    rescale_values = True
    if str(pixel_values_units) == 'mm2/s':
        rescale_values = False

    if rescale_values:
        for i in range(num_slices):
            pixel_arrays[:, :, i] = pixel_arrays[:, :, i] * rwv_slope[i] + rwv_intercept[i]
    
    # Generate x, y, z coordinates using the DICOM ImageOrientationPatient and ImagePositionPatient
    row_spacing, col_spacing = pixel_spacing  # XY pixel spacing (in mm)
    
    # Image orientation vectors
    orientation_x = np.array(image_orientation[:3])  # Direction of the x-axis in patient coordinates
    orientation_y = np.array(image_orientation[3:])  # Direction of the y-axis in patient coordinates

    # Generate x and y coordinates for each pixel in the 2D slices
    x_coords = np.arange(slice_shape[1]) * col_spacing
    y_coords = np.arange(slice_shape[0]) * row_spacing
    
    # Create a meshgrid for x and y coordinates
    x_grid, y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')

    # Initialize an array to store voxel positions (x, y, z, ADC values)
    voxel_positions = np.zeros((slice_shape[0] * slice_shape[1] * num_slices, 4))

    # Iterate over slices and calculate the physical positions in 3D space
    for i in range(num_slices):
        # Get the slice position (upper-left corner in patient coordinates)
        slice_origin = image_positions[i]

        # Compute the 3D coordinates for each voxel in the slice
        x_phys = slice_origin[0] + x_grid * orientation_x[0] + y_grid * orientation_y[0]
        y_phys = slice_origin[1] + x_grid * orientation_x[1] + y_grid * orientation_y[1]
        z_phys = slice_origin[2] + x_grid * orientation_x[2] + y_grid * orientation_y[2]

        # Flatten the grids and store the positions and ADC values
        start_idx = i * slice_shape[0] * slice_shape[1]
        end_idx = (i + 1) * slice_shape[0] * slice_shape[1]
        voxel_positions[start_idx:end_idx, 0] = x_phys.ravel()
        voxel_positions[start_idx:end_idx, 1] = y_phys.ravel()
        voxel_positions[start_idx:end_idx, 2] = z_phys.ravel()
        voxel_positions[start_idx:end_idx, 3] = pixel_arrays[:, :, i].ravel()

    # Filter out negative values
    if set_negative_to_value is not None and filter_out_negatives:
        raise ValueError("Cannot set both 'filter_out_negatives=True' and 'set_negative_to_value'. Choose one.")
    elif set_negative_to_value is not None:
        voxel_positions[voxel_positions[:, 3] < 0, 3] = set_negative_to_value
    elif filter_out_negatives:
        voxel_positions = voxel_positions[voxel_positions[:, 3] >= 0]



    return voxel_positions
