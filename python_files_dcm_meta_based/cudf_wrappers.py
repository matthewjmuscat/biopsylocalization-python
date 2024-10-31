import GPUtil
import pandas as pd
import cudf

def get_available_gpu_memory():
    """Returns available GPU memory in bytes."""
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Get the first available GPU
        return gpu.memoryFree * 1024 ** 2  # Convert MB to bytes
    else:
        raise ValueError("No GPU found")

def estimate_row_memory(df):
    """Estimate memory footprint of a single row in the DataFrame."""
    mem_usage_per_row = df.memory_usage(deep=True).sum() / len(df)
    return mem_usage_per_row

def calculate_chunk_size(df, safety_factor=0.8):
    """Calculate chunk size based on available GPU memory and DataFrame's row size."""
    available_memory = get_available_gpu_memory()  # In bytes
    row_memory = estimate_row_memory(df)  # Memory per row in bytes
    
    # Safety factor to avoid overloading GPU memory
    effective_memory = available_memory * safety_factor
    
    # Calculate number of rows that fit into the available GPU memory
    chunk_size = int(effective_memory // row_memory)
    return chunk_size

def process_in_chunks(df, func, safety_factor=0.8):
    """Process DataFrame in chunks based on available GPU memory."""
    chunk_size = calculate_chunk_size(df, safety_factor)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    result_chunks = []
    
    for chunk in chunks:
        cudf_chunk = cudf.DataFrame.from_pandas(chunk)  # Convert to cuDF
        cudf_result_chunk = func(cudf_chunk)  # Process chunk
        result_chunks.append(cudf_result_chunk.to_pandas())  # Convert back to pandas
    
    return pd.concat(result_chunks, ignore_index=True)

# Example custom processing function on cuDF
def custom_function_on_cudf(cudf_chunk):
    cudf_chunk['result'] = cudf_chunk['column1'] + cudf_chunk['column2']
    return cudf_chunk



def cudf_wrapper(func, df, chunk_size=None, **kwargs):
    # If chunking is needed
    if chunk_size:
        return process_in_chunks(df, func)
    
    # If no chunking, just convert to cuDF
    cudf_df = cudf.DataFrame.from_pandas(df)
    
    # Call the original function with cuDF DataFrame
    cudf_result = func(cudf_df, **kwargs)
    
    # Convert the result back to pandas
    return cudf_result.to_pandas()




# Example cuDF function
def example_cudf_function(cudf_df):
    # Perform operations with cuDF
    cudf_df['result'] = cudf_df['column1'] + cudf_df['column2']
    return cudf_df

# Wrapping the function
df_result = cudf_wrapper(example_cudf_function, your_dataframe, chunk_size=10000)


# Example usage with chunking
df_result = process_in_chunks(your_dataframe, custom_function_on_cudf)











###############################











import GPUtil
import pandas as pd
import cudf

def get_available_gpu_memory():
    """Returns available GPU memory in bytes."""
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Get the first available GPU
        return gpu.memoryFree * 1024 ** 2  # Convert MB to bytes
    else:
        raise ValueError("No GPU found")

def estimate_row_memory(df):
    """Estimate memory footprint of a single row in the DataFrame."""
    mem_usage_per_row = df.memory_usage(deep=True).sum() / len(df)
    return mem_usage_per_row

def calculate_chunk_size(df, safety_factor=0.8):
    """Calculate chunk size based on available GPU memory and DataFrame's row size."""
    available_memory = get_available_gpu_memory()  # In bytes
    row_memory = estimate_row_memory(df)  # Memory per row in bytes
    
    # Safety factor to avoid overloading GPU memory
    effective_memory = available_memory * safety_factor
    
    # Calculate number of rows that fit into the available GPU memory
    chunk_size = int(effective_memory // row_memory)
    return chunk_size

def convert_to_cudf(df):
    """Convert a pandas DataFrame to cuDF if a GPU is available."""
    try:
        cudf_df = cudf.DataFrame.from_pandas(df)
        return cudf_df
    except Exception as e:
        print(f"Error converting to cuDF: {e}")
        return df  # Fall back to pandas if there's an error

def convert_to_pandas(cudf_df):
    """Convert a cuDF DataFrame back to pandas."""
    try:
        pandas_df = cudf_df.to_pandas()
        return pandas_df
    except Exception as e:
        print(f"Error converting back to pandas: {e}")
        return cudf_df  # Fall back to cuDF if there's an error

def process_in_chunks_with_gpu(df, func, safety_factor=0.8):
    """Process a pandas DataFrame in chunks using GPU (cuDF) to handle memory limits."""
    chunk_size = calculate_chunk_size(df, safety_factor)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
    result_chunks = []

    for chunk in chunks:
        cudf_chunk = convert_to_cudf(chunk)  # Convert each chunk to cuDF
        cudf_result_chunk = func(cudf_chunk)  # Apply the function on cuDF chunk
        result_chunks.append(convert_to_pandas(cudf_result_chunk))  # Convert result back to pandas

    return pd.concat(result_chunks, ignore_index=True)

# Define the wrapper function
def gpu_accelerated_function(func):
    """A generic wrapper to execute a function on cuDF DataFrames, with chunking for large dataframes."""
    def wrapped_function(*args, **kwargs):
        # Extract the DataFrame arguments from the function and apply chunking
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                # Process this DataFrame in chunks if it's too large
                args[i] = process_in_chunks_with_gpu(arg, func)
        return func(*args, **kwargs)
    
    return wrapped_function

# Example: Apply the wrapper to your global dosimetry function
@gpu_accelerated_function
def global_dosimetry_values_dataframe_builder(master_structure_reference_dict, bx_ref, all_ref_key, dose_ref):
    # Your function logic here remains the same
    ...
