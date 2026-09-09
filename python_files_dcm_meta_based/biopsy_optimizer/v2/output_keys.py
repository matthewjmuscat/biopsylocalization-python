"""Stable optimizer-v2 output keys shared by runtime and artifact contracts."""

TARGET_DIL_OPTIMIZER_V2_LANE_NAME = "target_dil_optimizer_v2"
TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 summary dataframe"
)
TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 ranked candidates dataframe"
)
TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 tested candidates dataframe"
)
TARGET_DIL_OPTIMIZER_V2_STAGE_BOUNDARY_RENDER_JOBS_KEY = (
    "Biopsy optimization - Target DIL optimizer v2 stage boundary render jobs"
)
TARGET_DIL_OPTIMIZER_V2_DOWNSTREAM_MC_SOURCE_DF_KEY = (
    "Tissue class - Global tissue by structure statistics"
)


__all__ = [
    "TARGET_DIL_OPTIMIZER_V2_DOWNSTREAM_MC_SOURCE_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_LANE_NAME",
    "TARGET_DIL_OPTIMIZER_V2_RANKED_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_STAGE_BOUNDARY_RENDER_JOBS_KEY",
    "TARGET_DIL_OPTIMIZER_V2_SUMMARY_DF_KEY",
    "TARGET_DIL_OPTIMIZER_V2_TESTED_DF_KEY",
]
