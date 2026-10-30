"""Execution backend for sector tasks.

Three backends, one interface:

``process``  real CPU parallelism on stock CPython (the GIL is bypassed).
``thread``   parallelism on free-threaded builds, and useful overlap when the
             work is dominated by database round trips.
``serial``   everything inline; always available, always correct.

The pool is created once per run and reused across rounds.  ``spawn`` is used
explicitly rather than ``fork``: forking a process that already holds psycopg
connections is unsafe, and ``fork`` does not exist on Windows.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pickle import PicklingError

from src.sector_task import (
    close_all_connections,
    simulate_sector_task,
    warm_worker,
)

# Failures that mean "this backend cannot be used here", as opposed to a bug in
# the simulation code (which must propagate, not silently fall back).
POOL_FAILURES = (BrokenProcessPool, PicklingError, EOFError, OSError, ImportError)


class SectorPool:
    def __init__(self, backend: str, workers: int, logger=None):
        self.requested_backend = backend
        self.workers = max(1, int(workers))
        self.backend = "serial"
        self.fallback_reason = ""
        self._pool = None
        self._logger = logger
        self._warm: list = []

        if backend not in ("thread", "process") or self.workers <= 1:
            return

        try:
            if backend == "process":
                context = multiprocessing.get_context("spawn")
                # No ``finalizer``: ProcessPoolExecutor does not support it.
                # Worker connections are closed when the worker process exits.
                self._pool = ProcessPoolExecutor(
                    max_workers=self.workers,
                    mp_context=context,
                )
            else:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.workers,
                    thread_name_prefix="sector",
                )
            self.backend = backend
        except Exception as exc:  # noqa: BLE001 - degrade, never abort a run
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            self._pool = None
            self._log(
                f"[parallel] could not start a {backend} pool "
                f"({self.fallback_reason}); running single-threaded"
            )

    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        return self._pool is not None

    # ------------------------------------------------------------------ #
    def warm(self, db_config: dict | None = None) -> None:
        """Start the workers now, without blocking the caller.

        Process start-up (spawn + imports) and the workers' own PostgreSQL
        connections are then paid while the orchestrator is still busy, instead
        of delaying the first round.  Threads start instantly, so only the
        process backend is worth warming.
        """
        if self._pool is None or self.backend != "process":
            return
        for _ in range(self.workers):
            self._warm.append(self._pool.submit(warm_worker, db_config))

    def _finish_warm(self) -> None:
        """Wait for the warm-up tasks; propagate only pool-level failures."""
        futures, self._warm = self._warm, []
        for future in futures:
            try:
                future.result()
            except POOL_FAILURES:
                raise
            except Exception:  # noqa: BLE001 - a real task will report it
                pass

    def map(self, tasks):
        if self._pool is None:
            raise RuntimeError("SectorPool.map() called on an inactive pool")
        self._finish_warm()
        return list(self._pool.map(simulate_sector_task, tasks))

    def close(self) -> None:
        pool, self._pool = self._pool, None
        self._warm = []
        if pool is not None:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except TypeError:  # pragma: no cover - Python < 3.9
                pool.shutdown(wait=True)
        close_all_connections()

    # ------------------------------------------------------------------ #
    def _log(self, message: str) -> None:
        if self._logger is not None:
            self._logger(message)

    def __enter__(self) -> "SectorPool":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        state = "active" if self.active else "inactive"
        return f"SectorPool({self.backend}, workers={self.workers}, {state})"
