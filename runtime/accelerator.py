"""Accelerated nearest-enemy search: numpy first, pure Python last.

The sector engine spends most of its time answering "who is the closest living
enemy?" for every agent, every tick.  That question is one batched operation per
group, so it can be answered by numpy instead of a Python loop.

Backends, in the order they are tried:

``numpy``  the same query in exact float64 on the CPU.
``python`` :class:`utils.spatial.SpatialIndex`, the reference path.

Determinism is not negotiable: a seeded run must produce the same rows no matter
which backend ran it, so a batched backend only supplies indices and every
distance is rebuilt from the original float64 positions
(:func:`exact_distances`).  A backend therefore cannot change a result, only
the speed.

Everything degrades instead of failing: a missing ``numpy`` module falls back
to the scalar path.
"""

from __future__ import annotations

import math
import threading

from config import simulation_config as cfg


def exact_distances(queries, targets, indices) -> list[float]:
    """Turn chosen indices into the distances the scalar path would report.

    ``math.sqrt`` on the same float64 expression as ``SpatialIndex.nearest``
    keeps every downstream decision (``can_reach``, ``hit_chance``,
    ``planned_move``) bit-identical across backends.
    """
    out: list[float] = []
    for (qx, qy), index in zip(queries, indices):
        if index < 0:
            out.append(float("inf"))
            continue
        tx, ty = targets[index]
        dx = tx - qx
        dy = ty - qy
        out.append(math.sqrt(dx * dx + dy * dy))
    return out


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class PythonAccelerator:
    """Present so callers can treat "no acceleration" as a normal backend."""

    name = "python"

    def nearest_cross(self, queries, targets):
        if not queries or not targets:
            return [-1] * len(queries), [float("inf")] * len(queries)
        # Kept simple on purpose: this path only runs when numpy is missing.
        indices = []
        for qx, qy in queries:
            best_i, best_d2 = -1, float("inf")
            for index, (tx, ty) in enumerate(targets):
                dx = tx - qx
                dy = ty - qy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_i, best_d2 = index, d2
            indices.append(best_i)
        return indices, exact_distances(queries, targets, indices)


class NumpyAccelerator:
    name = "numpy"

    def __init__(self) -> None:
        from utils import spatial_batch

        self._batch = spatial_batch
        self._scalar = PythonAccelerator()

    def nearest_cross(self, queries, targets):
        if not queries or not targets:
            return [-1] * len(queries), [float("inf")] * len(queries)
        if len(queries) * len(targets) < int(cfg.ACCEL_MIN_PAIRS):
            # A handful of agents is cheaper to compare in Python than to hand
            # to numpy; this is also the end-game tail of most runs.
            return self._scalar.nearest_cross(queries, targets)
        indices = self._batch.nearest_indices(queries, targets)
        return indices, exact_distances(queries, targets, indices)


# --------------------------------------------------------------------------- #
# Probing and the process-wide singleton
# --------------------------------------------------------------------------- #
class AccelPlan:
    """The resolved acceleration choice for one run."""

    __slots__ = ("mode",)

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def describe(self) -> str:
        if self.mode == "numpy":
            return "numpy (CPU)"
        return "pure Python (scalar grid)"


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - a broken install is the same as absent
        return False


_global_lock = threading.RLock()
_global_accel = None
_global_mode = ""
_global_requested = ""
_global_plan: AccelPlan | None = None


def probe(requested: str | None = None) -> AccelPlan:
    """Resolve ``auto|off|numpy`` into a mode."""
    raw = (cfg.ACCEL if requested is None else requested) or "auto"
    raw = str(raw).strip().lower()

    if raw == "off":
        return AccelPlan("python")

    return AccelPlan("numpy" if _numpy_available() else "python")


def get() -> object:
    """The process-wide accelerator, created once."""
    global _global_accel
    with _global_lock:
        if _global_accel is None:
            if _global_mode == "numpy":
                _global_accel = NumpyAccelerator()
            else:
                _global_accel = PythonAccelerator()
        return _global_accel


def mode() -> str:
    """The effective backend right now ("python" until configured)."""
    return _global_mode or "python"


def requested() -> str:
    """The mode this process was configured with ("auto" until configured)."""
    return _global_requested


def configure(requested: str | None = None) -> AccelPlan:
    """Fix the backend for this process.  Safe to call more than once."""
    global _global_mode, _global_accel, _global_plan, _global_requested
    plan = probe(requested)
    raw = (cfg.ACCEL if requested is None else requested) or "auto"
    key = str(raw).strip().lower()
    with _global_lock:
        if _global_plan is not None and key == _global_requested:
            return _global_plan
        close()
        _global_requested = key
        _global_mode = plan.mode
        _global_plan = plan
    return plan


def close() -> None:
    """Drop the backend (idempotent); the next ``get()`` rebuilds it."""
    global _global_accel
    with _global_lock:
        _global_accel = None
