"""Spatial indexing for the hot nearest-enemy query.

The sector engine asks "who is the closest living enemy?" for every agent, every
tick, twice (movement and combat).  A brute-force scan is O(n*m); this module
answers the same question from a uniform grid in roughly O(1) expected time.

Correctness is deliberately exact: the winner is chosen by ``(dist_sq,
agent_id)``, which matches the old brute-force loop for the canonical
(ascending-id) agent order used by the database layer.  :func:`brute_force`
is kept here so tests can prove the index never disagrees.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

_INF = float("inf")


class SpatialIndex:
    """Uniform-grid index over living agents.

    ``bounds`` is the walkable box of the sector (``(min_x, min_y, max_x,
    max_y)``).  Supplying it keeps the grid aligned with the arena; without it
    the agents' own extent is used.  Only living agents are indexed: the engine
    never targets a corpse.
    """

    __slots__ = (
        "agents",
        "cell_size",
        "min_x",
        "min_y",
        "cols",
        "rows",
        "cells",
    )

    def __init__(
        self,
        agents: Iterable,
        bounds: Sequence[float] | None = None,
        cell_size: float | None = None,
    ):
        self.agents = [agent for agent in agents if getattr(agent, "is_alive", True)]

        if not self.agents:
            self.cell_size = 1.0
            self.min_x = self.min_y = 0.0
            self.cols = self.rows = 1
            self.cells: dict[tuple[int, int], list] = {}
            return

        if bounds is not None:
            min_x, min_y, max_x, max_y = (float(value) for value in bounds)
        else:
            xs = [agent.x_pos for agent in self.agents]
            ys = [agent.y_pos for agent in self.agents]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)
        count = len(self.agents)

        if cell_size is None:
            # Aim for a handful of agents per cell.  Larger cells mean fewer
            # rings to scan; smaller cells mean fewer candidates per ring.
            area = max(1.0, span_x * span_y)
            cell_size = max(1.0, 2.0 * math.sqrt(area / max(1, count)))

        self.cell_size = float(cell_size)
        self.min_x = min_x
        self.min_y = min_y
        self.cols = max(1, int(math.ceil(span_x / self.cell_size)) + 1)
        self.rows = max(1, int(math.ceil(span_y / self.cell_size)) + 1)

        cells: dict[tuple[int, int], list] = {}
        for agent in self.agents:
            cells.setdefault(
                self._clamped_cell(agent.x_pos, agent.y_pos), []
            ).append(agent)
        self.cells = cells

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def nearest(
        self, x: float, y: float, exclude_id: int | None = None
    ) -> tuple[object | None, float]:
        """Return ``(closest living agent, distance)``.

        Ties are broken by the smallest ``agent_id`` so a parallel shard and the
        serial loop always agree.  Returns ``(None, inf)`` for an empty index.
        """
        if not self.agents:
            return None, _INF

        cx, cy = self._cell(x, y)
        if not (0 <= cx < self.cols and 0 <= cy < self.rows):
            # A query outside the grid is rare (agents are bounded by their
            # sector).  Fall back to the reference implementation rather than
            # trusting the ring lower bound.
            return brute_force(self.agents, x, y, exclude_id=exclude_id)

        best = None
        best_sq = _INF
        best_id = _INF
        max_ring = max(cx, cy, self.cols - 1 - cx, self.rows - 1 - cy)

        for ring in range(max_ring + 1):
            for dx, dy in _ring_offsets(ring):
                gx = cx + dx
                gy = cy + dy
                if not (0 <= gx < self.cols and 0 <= gy < self.rows):
                    continue
                for agent in self.cells.get((gx, gy), ()):
                    if agent.agent_id == exclude_id:
                        continue
                    dx_pos = agent.x_pos - x
                    dy_pos = agent.y_pos - y
                    dist_sq = dx_pos * dx_pos + dy_pos * dy_pos
                    if dist_sq < best_sq or (
                        dist_sq == best_sq and agent.agent_id < best_id
                    ):
                        best = agent
                        best_sq = dist_sq
                        best_id = agent.agent_id

            if best is not None:
                # Conservative: every unvisited cell starts at least this far
                # away (a half-open cell at Chebyshev ring r begins at
                # (r-1)*cell_size from a point inside the centre cell).
                lower = max(0.0, (ring - 1) * self.cell_size)
                if best_sq <= lower * lower:
                    break

        if best is None:
            return None, _INF
        return best, math.sqrt(best_sq)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return self._cell_index(x - self.min_x), self._cell_index(y - self.min_y)

    def _clamped_cell(self, x: float, y: float) -> tuple[int, int]:
        """Cell for an agent, clamped into the grid.

        Agents can sit outside a caller-supplied ``bounds`` (tight sector
        padding, for example). Clamping keeps them in a real cell so they stay
        findable instead of being silently dropped.
        """
        cx, cy = self._cell(x, y)
        return (
            min(max(cx, 0), self.cols - 1),
            min(max(cy, 0), self.rows - 1),
        )

    def _cell_index(self, offset: float) -> int:
        return int(math.floor(offset / self.cell_size))

    def __len__(self) -> int:
        return len(self.agents)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SpatialIndex(agents={len(self.agents)}, grid={self.cols}x{self.rows}, "
            f"cell={self.cell_size:.1f})"
        )


def brute_force(
    agents: Iterable, x: float, y: float, exclude_id: int | None = None
) -> tuple[object | None, float]:
    """Reference nearest-enemy scan used by the tests and the fallback path."""
    best = None
    best_sq = _INF
    best_id = _INF
    for agent in agents:
        if not getattr(agent, "is_alive", True):
            continue
        if agent.agent_id == exclude_id:
            continue
        dx = agent.x_pos - x
        dy = agent.y_pos - y
        dist_sq = dx * dx + dy * dy
        if dist_sq < best_sq or (dist_sq == best_sq and agent.agent_id < best_id):
            best = agent
            best_sq = dist_sq
            best_id = agent.agent_id
    if best is None:
        return None, _INF
    return best, math.sqrt(best_sq)


def _ring_offsets(ring: int):
    """Cells whose Chebyshev distance from the centre is exactly ``ring``."""
    if ring == 0:
        return ((0, 0),)
    offsets = []
    for dx in range(-ring, ring + 1):
        offsets.append((dx, -ring))
        offsets.append((dx, ring))
    for dy in range(-ring + 1, ring):
        offsets.append((-ring, dy))
        offsets.append((ring, dy))
    return offsets
