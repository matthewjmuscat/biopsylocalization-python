_LAZY_EXPORTS = {
    "NonBiopsyStructurePreprocessingConfig": "preprocessing.structure_processing.non_biopsy_structure_processing",
    "STRUCTURE_PREPROCESSING_TIMINGS_DF_KEY": "preprocessing.structure_processing.non_biopsy_structure_processing",
    "preprocess_non_biopsy_structure": "preprocessing.structure_processing.non_biopsy_structure_processing",
    "STRUCTURE_PREPROCESSING_VALIDATION_DF_KEY": "preprocessing.structure_processing.validation",
    "append_non_biopsy_structure_validation_result": "preprocessing.structure_processing.validation",
    "build_non_biopsy_structure_modular_snapshot": "preprocessing.structure_processing.validation",
    "capture_non_biopsy_structure_processing_snapshot": "preprocessing.structure_processing.validation",
    "compare_non_biopsy_structure_processing_snapshots": "preprocessing.structure_processing.validation",
    "pull_raw_structure_contours_for_cohort": "preprocessing.structure_processing.raw_contour_pulling",
    "pull_raw_structure_contour_for_structure": "preprocessing.structure_processing.raw_contour_pulling",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value

__all__ = [
    "NonBiopsyStructurePreprocessingConfig",
    "STRUCTURE_PREPROCESSING_TIMINGS_DF_KEY",
    "STRUCTURE_PREPROCESSING_VALIDATION_DF_KEY",
    "append_non_biopsy_structure_validation_result",
    "build_non_biopsy_structure_modular_snapshot",
    "capture_non_biopsy_structure_processing_snapshot",
    "compare_non_biopsy_structure_processing_snapshots",
    "preprocess_non_biopsy_structure",
    "pull_raw_structure_contour_for_structure",
    "pull_raw_structure_contours_for_cohort",
]
