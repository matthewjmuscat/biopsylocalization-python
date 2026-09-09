"""Worker-owned runtime resources for standalone patient execution."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import starmap as itertools_starmap
from typing import Any


class SequentialWorkerPool:
    """Small ``multiprocessing.Pool`` map surface executed in the worker process.

    The standalone parent already isolates patients in separate processes. This
    adapter preserves the legacy stage call contract without creating nested
    process pools or sharing mutable resources between patients.
    """

    def map(
        self,
        function: Callable[[Any], Any],
        iterable: Iterable[Any],
        chunksize: int | None = None,
    ) -> list[Any]:
        del chunksize
        return list(map(function, iterable))

    def starmap(
        self,
        function: Callable[..., Any],
        iterable: Iterable[Iterable[Any]],
        chunksize: int | None = None,
    ) -> list[Any]:
        del chunksize
        return list(itertools_starmap(function, iterable))


__all__ = ["SequentialWorkerPool"]