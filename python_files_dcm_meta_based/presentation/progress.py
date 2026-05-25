"""Presentation-neutral progress and event contracts.

Scientific patient modules should depend on these small interfaces instead of
Rich, GUI widgets, or legacy live-display objects. Legacy wrappers can adapt
these calls to Rich while headless validation can use the null implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Structured status event emitted by a patient-stage function."""

    event_type: str
    message: str = ""
    level: str = "INFO"
    phase: str | None = None
    patient_uid: str | None = None
    stage_name: str | None = None
    structure_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_type", str(self.event_type).strip())
        if self.event_type == "":
            raise ValueError("event_type cannot be empty")
        object.__setattr__(self, "message", str(self.message))
        object.__setattr__(self, "level", str(self.level).strip().upper() or "INFO")
        object.__setattr__(self, "details", dict(self.details or {}))


class ProgressTask(Protocol):
    """Small task handle independent of any concrete presentation library."""

    def update(self,
               *,
               advance: float | None = None,
               completed: float | None = None,
               total: float | None = None,
               description: str | None = None,
               visible: bool | None = None,
               details: Mapping[str, Any] | None = None) -> None:
        ...

    def finish(self,
               *,
               message: str = "",
               details: Mapping[str, Any] | None = None) -> None:
        ...


class ProgressSink(Protocol):
    """Presentation-neutral sink for patient-stage status events."""

    def emit(self, event: ProgressEvent) -> None:
        ...

    def task(self,
             description: str,
             *,
             total: float | None = None,
             patient_uid: str | None = None,
             stage_name: str | None = None,
             details: Mapping[str, Any] | None = None) -> ProgressTask:
        ...


@dataclass(slots=True)
class NullProgressTask:
    """No-op task used for headless execution and tests."""

    description: str = ""
    total: float | None = None
    completed: float = 0.0
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def update(self,
               *,
               advance: float | None = None,
               completed: float | None = None,
               total: float | None = None,
               description: str | None = None,
               visible: bool | None = None,
               details: Mapping[str, Any] | None = None) -> None:
        if advance is not None:
            self.completed += float(advance)
        if completed is not None:
            self.completed = float(completed)
        if total is not None:
            self.total = float(total)
        if description is not None:
            self.description = str(description)
        if visible is not None:
            self.visible = bool(visible)
        if details:
            self.metadata.update(dict(details))

    def finish(self,
               *,
               message: str = "",
               details: Mapping[str, Any] | None = None) -> None:
        if self.total is not None:
            self.completed = float(self.total)
        self.visible = False
        if message:
            self.metadata["finish_message"] = str(message)
        if details:
            self.metadata.update(dict(details))


class NullProgressSink:
    """No-op sink for batch/headless patient execution."""

    def emit(self, event: ProgressEvent) -> None:
        return None

    def task(self,
             description: str,
             *,
             total: float | None = None,
             patient_uid: str | None = None,
             stage_name: str | None = None,
             details: Mapping[str, Any] | None = None) -> NullProgressTask:
        metadata = dict(details or {})
        if patient_uid is not None:
            metadata["patient_uid"] = patient_uid
        if stage_name is not None:
            metadata["stage_name"] = stage_name
        return NullProgressTask(
            description=str(description),
            total=total,
            metadata=metadata,
        )

    def info(self, message: str, **details: Any) -> None:
        self.emit(ProgressEvent("info", message=message, details=details))

    def warning(self, message: str, **details: Any) -> None:
        self.emit(ProgressEvent("warning", message=message, level="WARNING", details=details))


def coerce_progress_sink(progress_sink: ProgressSink | None) -> ProgressSink:
    """Return a usable sink, defaulting to a no-op implementation."""
    if progress_sink is None:
        return NullProgressSink()
    return progress_sink


class LegacyNullProgress:
    """Duck-typed no-op replacement for Rich Progress objects.

    This is for compatibility adapters around old cohort-facing functions. New
    patient science should prefer ``ProgressSink``/``ProgressTask`` directly.
    """

    def __init__(self) -> None:
        self._next_task_id = 0
        self.tasks: dict[int, dict[str, Any]] = {}

    def add_task(self,
                 description: str = "",
                 *,
                 total: float | None = None,
                 visible: bool = True,
                 **fields: Any) -> int:
        task_id = self._next_task_id
        self._next_task_id += 1
        self.tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0.0,
            "visible": visible,
            "fields": dict(fields),
        }
        return task_id

    def update(self,
               task_id: int,
               *,
               advance: float | None = None,
               completed: float | None = None,
               total: float | None = None,
               description: str | None = None,
               visible: bool | None = None,
               refresh: bool | None = None,
               **fields: Any) -> None:
        task = self.tasks.setdefault(int(task_id), {"completed": 0.0, "fields": {}})
        if advance is not None:
            task["completed"] = float(task.get("completed", 0.0)) + float(advance)
        if completed is not None:
            task["completed"] = float(completed)
        if total is not None:
            task["total"] = total
        if description is not None:
            task["description"] = description
        if visible is not None:
            task["visible"] = visible
        if fields:
            task.setdefault("fields", {}).update(fields)

    def remove_task(self, task_id: int) -> None:
        self.tasks.pop(int(task_id), None)

    def stop_task(self, task_id: int) -> None:
        self.update(task_id, visible=False)

    def start_task(self, task_id: int) -> None:
        self.update(task_id, visible=True)


class LegacyNullLiveDisplay:
    """Minimal no-op object with the methods used by legacy wrappers."""

    is_started = False

    def start(self, *args: Any, **kwargs: Any) -> None:
        self.is_started = True

    def stop(self, *args: Any, **kwargs: Any) -> None:
        self.is_started = False

    def refresh(self, *args: Any, **kwargs: Any) -> None:
        return None

    def update(self, *args: Any, **kwargs: Any) -> None:
        return None


class LegacyImportantInfoSink:
    """Collect text lines from legacy functions without requiring a Rich panel."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def add_text_line(self, text: str, live_display: Any = None) -> None:
        self.lines.append(str(text))
        if live_display is not None and hasattr(live_display, "refresh"):
            live_display.refresh()


@dataclass(slots=True)
class LegacyPresentationContext:
    """Compatibility bundle for old functions that still expect Rich-like args."""

    layout_groups: Any = None
    patients_progress: Any = field(default_factory=LegacyNullProgress)
    structures_progress: Any = field(default_factory=LegacyNullProgress)
    biopsies_progress: Any = field(default_factory=LegacyNullProgress)
    indeterminate_progress_main: Any = field(default_factory=LegacyNullProgress)
    indeterminate_progress_sub: Any = field(default_factory=LegacyNullProgress)
    completed_progress: Any = field(default_factory=LegacyNullProgress)
    completed_sections_progress: Any = field(default_factory=LegacyNullProgress)
    important_info: Any = field(default_factory=LegacyImportantInfoSink)
    live_display: Any = field(default_factory=LegacyNullLiveDisplay)

    @classmethod
    def null(cls) -> "LegacyPresentationContext":
        return cls()