"""Duck-typed adapters from neutral progress events to Rich-style objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .progress import NullProgressTask
from .progress import ProgressEvent
from .progress import ProgressTask


@dataclass(slots=True)
class RichProgressTask:
    """Task handle backed by a Rich ``Progress``-like object."""

    progress: Any
    task_id: Any
    live_display: Any = None
    refresh_on_update: bool = False

    def update(self,
               *,
               advance: float | None = None,
               completed: float | None = None,
               total: float | None = None,
               description: str | None = None,
               visible: bool | None = None,
               details: Mapping[str, Any] | None = None) -> None:
        update_kwargs: dict[str, Any] = {}
        if advance is not None:
            update_kwargs["advance"] = advance
        if completed is not None:
            update_kwargs["completed"] = completed
        if total is not None:
            update_kwargs["total"] = total
        if description is not None:
            update_kwargs["description"] = description
        if visible is not None:
            update_kwargs["visible"] = visible
        if details:
            update_kwargs.update(dict(details))
        self.progress.update(self.task_id, **update_kwargs)
        if self.refresh_on_update and self.live_display is not None:
            self.live_display.refresh()

    def finish(self,
               *,
               message: str = "",
               details: Mapping[str, Any] | None = None) -> None:
        self.update(visible=False, details=details)


@dataclass(slots=True)
class RichProgressSink:
    """Translate neutral progress events into Rich/logging side effects."""

    progress: Any = None
    live_display: Any = None
    important_info: Any = None
    runtime_logger: Any = None
    refresh_on_update: bool = False
    emitted_events: list[ProgressEvent] = field(default_factory=list)

    def emit(self, event: ProgressEvent) -> None:
        self.emitted_events.append(event)
        if self.runtime_logger is not None:
            self._emit_to_runtime_logger(event)
        if event.message and self.important_info is not None:
            self.important_info.add_text_line(event.message, self.live_display)

    def task(self,
             description: str,
             *,
             total: float | None = None,
             patient_uid: str | None = None,
             stage_name: str | None = None,
             details: Mapping[str, Any] | None = None) -> ProgressTask:
        if self.progress is None:
            metadata = dict(details or {})
            if patient_uid is not None:
                metadata["patient_uid"] = patient_uid
            if stage_name is not None:
                metadata["stage_name"] = stage_name
            return NullProgressTask(description=description, total=total, metadata=metadata)
        task_id = self.progress.add_task(description, total=total)
        return RichProgressTask(
            progress=self.progress,
            task_id=task_id,
            live_display=self.live_display,
            refresh_on_update=self.refresh_on_update,
        )

    def _emit_to_runtime_logger(self, event: ProgressEvent) -> None:
        details = dict(event.details or {})
        if event.patient_uid is not None:
            details.setdefault("patient_uid", event.patient_uid)
        if event.stage_name is not None:
            details.setdefault("stage_name", event.stage_name)
        if event.structure_id is not None:
            details.setdefault("structure_id", event.structure_id)
        if hasattr(self.runtime_logger, "log_event"):
            self.runtime_logger.log_event(
                level=event.level,
                event_type=event.event_type,
                phase=event.phase,
                message=event.message,
                details=details,
            )
        elif hasattr(self.runtime_logger, "checkpoint"):
            self.runtime_logger.checkpoint(
                event.event_type,
                event.message,
                patient_uid=event.patient_uid,
                details=details,
            )