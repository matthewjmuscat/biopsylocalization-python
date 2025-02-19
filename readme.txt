Note on installation:

Works with python version 3.11.11.

Works with CUDA toolkit 12.8 as well as with the following packages:

cubinlinker-cu11 = { index = "pypinvidia"}
ptxcompiler-cu11 = { index = "pypinvidia"}
cuda-python = ">=12.0"
rmm-cu12 = { index = "pypinvidia"}
cudf-cu12 = { index = "pypinvidia"}
cuspatial-cu12 = { index = "pypinvidia"} 
cupy-cuda12x = "*"

Note that as of Feb 2025, cubinlinker-cu11 and ptxcompiler-cu11 do not have a cu12 counterpart, however these seem to be compatible with cuda12. Note also that rmm-cu12 requires cuda-python = ">=12.0".
