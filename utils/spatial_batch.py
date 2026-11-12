"""Exact batched nearest-target search for the sector engine.

The scalar path in :mod:`utils.spatial` answers "who is closest?" one agent at a
time.  This module answers the same question for a whole group at once, which
is what a vectorised CPU (numpy) can actually use.

Exactness is the whole point: the winner is chosen by ``(dist_sq, list order)``
and the returned indices are then turned back into distances by
:func:`runtime.accelerator.exact_distances` using the *original* float64
positions, so a caller gets bit-identical numbers to ``SpatialIndex.nearest``
no matter which backend produced the indices.
"""

from __future__ import annotations

import numpy as np

from config import simulation_config as cfg


def nearest_indices(queries, targets) -> list[int]:
    """Index of the nearest target for every query, ``-1`` when there is none.

    ``queries``/``targets`` are sequences of ``(x, y)`` pairs in the caller's
    canonical order.  Ties resolve to the lowest position in ``targets``, which
    is what the scalar loop's strict ``<`` comparison does.

    The pair matrix is computed in tiles so a huge group cannot allocate a
    second copy of the world in temporaries.
    """
    count = len(queries)
    if count == 0 or not targets:
        return [-1] * count

    q = np.asarray(queries, dtype=np.float64).reshape(count, 2)
    t = np.asarray(targets, dtype=np.float64).reshape(len(targets), 2)

    out = np.empty(count, dtype=np.int64)
    step = max(1, int(cfg.NUMPY_TILE_ELEMS) // max(1, len(t)))
    for start in range(0, count, step):
        stop = min(count, start + step)
        dx = q[start:stop, 0][:, None] - t[None, :, 0]
        dy = q[start:stop, 1][:, None] - t[None, :, 1]
        d2 = dx * dx + dy * dy
        out[start:stop] = d2.argmin(axis=1)
    return out.tolist()
