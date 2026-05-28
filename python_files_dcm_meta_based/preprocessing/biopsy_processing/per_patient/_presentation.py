"""Presentation adapters for biopsy patient modules.

These helpers keep patient-science entrypoints callable without Rich/live UI
objects while still accepting the legacy progress objects supplied by cohort
wrappers during validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from presentation import LegacyImportantInfoSink
from presentation import LegacyNullLiveDisplay
from presentation import LegacyNullProgress


@dataclass(slots=True)
class PatientBiopsyPresentationBoundary:
    """Resolved presentation objects for legacy biopsy geometry helpers."""

    layout_groups: Any
    structures_progress: Any
    processing_structures_task: Any
    indeterminate_progress_sub: Any
    live_display: Any
    created_processing_structures_task: bool = False


def _build_null_layout_groups(*,
                              structures_progress: Any,
                              indeterminate_progress_sub: Any) -> tuple[Any, list[Any], Any, Any]:
    progress_group_info_list = [
        LegacyNullProgress(),
        LegacyNullProgress(),
        LegacyNullProgress(),
        structures_progress,
        LegacyNullProgress(),
        LegacyNullProgress(),
        LegacyNullProgress(),
        indeterminate_progress_sub,
        None,
    ]
    return (None, progress_group_info_list, LegacyImportantInfoSink(), None)


def resolve_patient_biopsy_presentation_boundary(*,
                                                 layout_groups: Any = None,
                                                 structures_progress: Any = None,
                                                 processing_structures_task: Any = None,
                                                 indeterminate_progress_sub: Any = None,
                                                 live_display: Any = None,
                                                 task_description: str = "Patient biopsy stage",
                                                 task_total: int | None = None) -> PatientBiopsyPresentationBoundary:
    """Return usable presentation shims for patient biopsy modules."""
    resolved_structures_progress = structures_progress or LegacyNullProgress()
    resolved_indeterminate_progress_sub = indeterminate_progress_sub or LegacyNullProgress()
    resolved_live_display = live_display or LegacyNullLiveDisplay()
    resolved_layout_groups = layout_groups or _build_null_layout_groups(
        structures_progress=resolved_structures_progress,
        indeterminate_progress_sub=resolved_indeterminate_progress_sub,
    )

    created_task = False
    resolved_processing_structures_task = processing_structures_task
    if resolved_processing_structures_task is None:
        resolved_processing_structures_task = resolved_structures_progress.add_task(
            task_description,
            total=task_total,
            visible=False,
        )
        created_task = True

    return PatientBiopsyPresentationBoundary(
        layout_groups=resolved_layout_groups,
        structures_progress=resolved_structures_progress,
        processing_structures_task=resolved_processing_structures_task,
        indeterminate_progress_sub=resolved_indeterminate_progress_sub,
        live_display=resolved_live_display,
        created_processing_structures_task=created_task,
    )