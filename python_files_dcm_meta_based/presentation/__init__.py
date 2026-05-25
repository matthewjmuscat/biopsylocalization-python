"""Presentation adapter surfaces for CLI, GUI, and headless execution."""

from .progress import LegacyImportantInfoSink
from .progress import LegacyNullLiveDisplay
from .progress import LegacyNullProgress
from .progress import LegacyPresentationContext
from .progress import NullProgressSink
from .progress import NullProgressTask
from .progress import ProgressEvent
from .progress import ProgressSink
from .progress import ProgressTask
from .progress import coerce_progress_sink
from .rich_adapter import RichProgressSink
from .rich_adapter import RichProgressTask

__all__ = [
    "LegacyImportantInfoSink",
    "LegacyNullLiveDisplay",
    "LegacyNullProgress",
    "LegacyPresentationContext",
    "NullProgressSink",
    "NullProgressTask",
    "ProgressEvent",
    "ProgressSink",
    "ProgressTask",
    "RichProgressSink",
    "RichProgressTask",
    "coerce_progress_sink",
]