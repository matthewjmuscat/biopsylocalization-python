"""Post-run utilities for completed localization outputs.

This package is intentionally outside the scientific runner. It loads completed
run manifests and derives secondary outputs without rerunning patient stages.
"""

from .cohort_assembly import DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH
from .cohort_assembly import POST_RUN_COHORT_ASSEMBLY_CONFIG_SCHEMA_VERSION
from .cohort_assembly import PostRunCohortAssemblyJobConfig
from .cohort_assembly import PostRunCohortAssemblyJobResult
from .cohort_assembly import format_post_run_cohort_assembly_summary
from .cohort_assembly import load_cohort_assembly_job_configs
from .cohort_assembly import load_patient_batch_result_from_manifest
from .cohort_assembly import resolve_patient_batch_manifest_path
from .cohort_assembly import run_post_run_cohort_assembly
from .cohort_assembly import run_post_run_cohort_assembly_jobs

__all__ = [
    "DEFAULT_COHORT_ASSEMBLY_CONFIG_PATH",
    "POST_RUN_COHORT_ASSEMBLY_CONFIG_SCHEMA_VERSION",
    "PostRunCohortAssemblyJobConfig",
    "PostRunCohortAssemblyJobResult",
    "format_post_run_cohort_assembly_summary",
    "load_cohort_assembly_job_configs",
    "load_patient_batch_result_from_manifest",
    "resolve_patient_batch_manifest_path",
    "run_post_run_cohort_assembly",
    "run_post_run_cohort_assembly_jobs",
]