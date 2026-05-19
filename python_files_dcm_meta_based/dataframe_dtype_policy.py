import numpy as np
import pandas


def _unique_ordered_columns(*column_name_groups):
    unique_column_names = []
    seen_column_names = set()

    for column_name_group in column_name_groups:
        for column_name in column_name_group:
            if column_name not in seen_column_names:
                unique_column_names.append(column_name)
                seen_column_names.add(column_name)

    return tuple(unique_column_names)


OPTIMIZER_V1_LOCATION_NUMERIC_COLUMNS = (
    'Test location (X)',
    'Test location (Y)',
    'Test location (Z)',
    'Test location to DIL centroid (X)',
    'Test location to DIL centroid (Y)',
    'Test location to DIL centroid (Z)',
    'Dist to DIL centroid',
    'Test location (Prostate centroid origin) (X)',
    'Test location (Prostate centroid origin) (Y)',
    'Test location (Prostate centroid origin) (Z)',
    'Dist to Prostate centroid',
    'Number of normal dist points contained',
    'Number of normal dist points tested',
    'Proportion of normal dist points contained',
    'X_plane_index',
    'Y_plane_index',
    'Z_plane_index',
    'Pt actually tested bool',
)

OPTIMIZER_V1_GUIDANCE_MAP_MAX_PLANES_NUMERIC_COLUMNS = (
    'Test location (Prostate centroid origin) (X)',
    'Test location (Prostate centroid origin) (Y)',
    'Test location (Prostate centroid origin) (Z)',
    'Proportion of normal dist points contained',
    'X_plane_index',
    'Y_plane_index',
    'Z_plane_index',
)

OPTIMIZER_V1_CUMULATIVE_PROJECTION_NUMERIC_COLUMNS = (
    'Coordinate 1',
    'Coordinate 2',
    'Proportion of normal dist points contained',
)

OPTIMIZER_V1_LOCATION_NEVER_CATEGORICAL_COLUMNS = OPTIMIZER_V1_LOCATION_NUMERIC_COLUMNS

OPTIMIZER_V1_GUIDANCE_MAP_MAX_PLANES_NEVER_CATEGORICAL_COLUMNS = _unique_ordered_columns(
    OPTIMIZER_V1_GUIDANCE_MAP_MAX_PLANES_NUMERIC_COLUMNS,
)

OPTIMIZER_V1_CUMULATIVE_PROJECTION_NEVER_CATEGORICAL_COLUMNS = _unique_ordered_columns(
    OPTIMIZER_V1_CUMULATIVE_PROJECTION_NUMERIC_COLUMNS,
)


def resolve_numeric_series(dataframe,
                           column_name,
                           errors='raise'):
    return pandas.to_numeric(dataframe[column_name], errors=errors)


def resolve_integer_series(dataframe,
                           column_name,
                           errors='raise'):
    numeric_series = pandas.to_numeric(dataframe[column_name], errors=errors)

    if numeric_series.isna().any():
        return numeric_series.astype('Int64')

    return numeric_series.astype(np.int64)


def resolve_numeric_columns(dataframe,
                            column_names,
                            errors='raise'):
    resolved_columns_dict = {}

    for column_name in column_names:
        resolved_columns_dict[column_name] = pandas.to_numeric(
            dataframe[column_name],
            errors=errors,
        )

    return pandas.DataFrame(resolved_columns_dict, index=dataframe.index)