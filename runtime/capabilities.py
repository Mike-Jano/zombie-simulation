"""What can this machine actually do?

The simulation is CPU-bound in pure Python, so the only backend that gives a
real speed-up on a stock CPython build is a *process* pool: the GIL stops
threads from running interpreted code in parallel.  On a free-threaded build
(``python3.14t``) threads are better - no pickling, lower memory - so the
choice is made at runtime instead of being hard-coded.

Everything here is standard library and architecture-neutral: no affinity
calls, no ``fork``, no third-party helpers.  If a probe is missing the result
degrades to the safe answer (one CPU, GIL on, serial backend).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from config import simulation_config as cfg

VALID_BACKENDS = ("auto", "serial", "thread", "process")


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #
def cpu_count() -> int:
    """Usable CPUs, honouring affinity and container limits where possible.

    ``os.process_cpu_count`` (Python 3.13+) reports the CPUs this process may
    actually run on, which is what matters in containers and on hybrid
    big.LITTLE parts where ``os.cpu_count`` overstates the usable cores.
    """
    probe = getattr(os, "process_cpu_count", None)
    if probe is not None:
        try:
            count = probe()
            if count:
                return max(1, int(count))
        except (OSError, NotImplementedError):
            pass
    return max(1, int(os.cpu_count() or 1))


def gil_enabled() -> bool:
    """False only on free-threaded CPython builds (3.13+, ``--disable-gil``)."""
    probe = getattr(sys, "_is_gil_enabled", None)
    if probe is None:
        return True
    try:
        return bool(probe())
    except Exception:  # pragma: no cover - defensive, never expected
        return True


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def resolve_workers(requested: int | None = None) -> int:
    """Turn the ``WORKERS`` knob (0 = auto) into a worker count."""
    raw = cfg.WORKERS if requested is None else requested
    try:
        raw = int(raw)
    except (TypeError, ValueError):
        raw = 0

    if raw > 0:
        # MAX_WORKERS is the documented ceiling, and every worker opens a
        # PostgreSQL connection, so an explicit value cannot raise it.
        workers = min(raw, int(cfg.MAX_WORKERS))
    else:
        workers = min(int(cfg.MAX_WORKERS), cpu_count())
    return max(1, workers)


def resolve_backend(requested: str | None = None, workers: int = 1) -> str:
    """Pick the widest backend this machine can actually use."""
    raw = cfg.WORKER_BACKEND if requested is None else requested
    raw = str(raw or "auto").strip().lower()

    if raw in ("", "auto"):
        if workers <= 1:
            return "serial"
        return "thread" if not gil_enabled() else "process"

    if raw in ("thread", "threads", "threading"):
        return "thread" if workers > 1 else "serial"
    if raw in ("process", "processes", "multiprocessing", "procs"):
        return "process" if workers > 1 else "serial"
    return "serial"


@dataclass(frozen=True)
class ParallelPlan:
    """The resolved parallel configuration for one run."""

    workers: int
    backend: str
    cpu_count: int
    gil_enabled: bool

    @property
    def is_parallel(self) -> bool:
        return self.backend in ("thread", "process") and self.workers > 1

    def describe(self) -> str:
        gil = "on" if self.gil_enabled else "off (free-threaded)"
        return (
            f"{self.backend} backend, {self.workers} worker(s), "
            f"{self.cpu_count} usable CPU(s), GIL {gil}"
        )


def resolve_plan(
    workers: int | None = None, backend: str | None = None
) -> ParallelPlan:
    count = cpu_count()
    resolved_workers = resolve_workers(workers)
    resolved_backend = resolve_backend(backend, resolved_workers)
    return ParallelPlan(
        workers=resolved_workers,
        backend=resolved_backend,
        cpu_count=count,
        gil_enabled=gil_enabled(),
    )


def resolve_event_shards(
    sector_count: int,
    workers: int,
    backend: str,
    requested: int | None = None,
) -> int:
    """How many per-tick shards one sector should split into.

    The outer pool and the inner shards share the same worker budget, so this
    fills spare capacity without oversubscribing: with 3 sectors and 12 workers
    each sector gets 4 shards.  Only the thread backend benefits (or is safe to
    over-subscribe); process and serial backends stay at one shard.

    ``requested`` is the config/CLI override: 0 or ``None`` means auto.
    """
    raw = cfg.EVENT_SHARDS if requested is None else requested
    try:
        raw = int(raw or 0)
    except (TypeError, ValueError):
        raw = 0

    mode = str(cfg.EVENT_PARALLEL or "auto").strip().lower()
    if mode in ("off", "none", "false", "0"):
        return 1
    if backend != "thread":
        return 1

    sectors = max(1, int(sector_count or 1))
    budget = max(1, int(workers or 1))
    if raw > 0:
        # A pinned count is still bounded by the worker budget: 100k shards
        # would otherwise mean 100k threads for one sector.
        return max(1, min(raw, budget))

    outer = min(sectors, budget)
    return max(1, budget // outer)
