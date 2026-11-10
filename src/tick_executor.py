"""Execution strategy for one sector's per-tick event phases.

The engine is written against this tiny interface:

* ``SerialTickExecutor`` - the reference path; one chunk after another.
* ``ThreadTickExecutor``  - chunks run concurrently.  Only useful on a
  free-threaded CPython build (or when the work is I/O-bound), because the GIL
  serialises interpreted bytecode on a normal build.

Chunking never affects results: workers produce per-agent proposals and the
engine merges them in canonical (ascending ``agent_id``) order.  That also
means a caller may choose any number of chunks without changing a seeded run.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor


def chunked(items, parts: int) -> list[list]:
    """Split ``items`` into at most ``parts`` contiguous chunks.

    Contiguous chunks preserve agent order, which keeps the merge canonical
    without an extra sort.
    """
    values = list(items)
    if not values:
        return []
    parts = max(1, int(parts or 1))
    if parts == 1 or len(values) <= 1:
        return [values]
    size = max(1, int(math.ceil(len(values) / parts)))
    return [values[start : start + size] for start in range(0, len(values), size)]


class SerialTickExecutor:
    """Run every chunk inline on the calling thread."""

    def map(self, fn, chunks):
        return [fn(chunk) for chunk in chunks]

    def close(self) -> None:
        return None


class ThreadTickExecutor:
    """Run chunks on a small thread pool.

    ``max_workers`` is deliberately independent of the sector count: the
    simulation decides how much of its global worker budget this sector gets.
    """

    def __init__(self, max_workers: int = 1):
        self.max_workers = max(1, int(max_workers))
        self._pool = None
        if self.max_workers > 1:
            self._pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="event",
            )

    @property
    def active(self) -> bool:
        return self._pool is not None

    def map(self, fn, chunks):
        if self._pool is None:
            return [fn(chunk) for chunk in chunks]
        return list(self._pool.map(fn, chunks))

    def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)


def make_executor(shards: int):
    """Pick the executor for a shard count (1 = serial reference)."""
    if int(shards or 1) > 1:
        return ThreadTickExecutor(int(shards))
    return SerialTickExecutor()
