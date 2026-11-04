"""Grid-based sectoring.

The map is cut into a rectangular grid, one cell per sector. Every agent lands
in exactly one cell, and each sector gets its own walkable bounding box, so only
a subset of the world is ever resident in memory. A grid is O(n) and, unlike
k-means, its cost does not grow with the square of the population.

The number of cells is driven by the population alone:

    max_sectors = max(1, ceil(initial_agents / AGENTS_PER_SECTOR))
    k = clamp(ceil(living_agents / AGENTS_PER_SECTOR), 1, max_sectors)

Fewer, larger sectors are deliberate. Sectors are the unit of parallel work,
but they also decide who can ever meet: a finely split map leaves sectors
containing only humans or only zombies, those sectors cannot do anything, and
the run ends with both sides alive. A coarse split keeps the two sides in the
same arena, which is what lets a run reach a real conclusion.

The map itself is sized from the starting population by
:func:`map_bounds_for`, so agent density stays roughly constant as a run grows.

Every sector is also guaranteed to hold both types (mono-type clusters are
merged into the nearest mixed one), so no parallel work is ever spent on a
sector that cannot fight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from config import simulation_config as cfg

Point = tuple[float, float]
Bounds = tuple[float, float, float, float]


def map_bounds_for(agent_count: int) -> Bounds:
    """The -/+ extent of the map for a run with ``agent_count`` agents.

    Area grows linearly with the population, so average spacing (and therefore
    how long a chase takes) stays roughly the same from a tiny run to a huge
    one. There is deliberately no upper clamp.
    """
    total = max(0, int(agent_count))
    half = max(
        int(cfg.MAP_MIN_HALF_EXTENT),
        int(math.ceil(math.sqrt(total * cfg.MAP_AREA_PER_AGENT) / 2.0)),
    )
    return (-float(half), -float(half), float(half), float(half))


def max_sectors_for(agent_count: int) -> int:
    """Upper bound on sectors for a run that started with ``agent_count``."""
    total = max(0, int(agent_count))
    if total <= 0:
        return 1
    return max(1, math.ceil(total / max(1, int(cfg.AGENTS_PER_SECTOR))))


@dataclass
class Sector:
    index: int
    agent_ids: list[int] = field(default_factory=list)
    bounds: Bounds = field(default_factory=lambda: map_bounds_for(0))


def resolve_sector_count(
    agent_count: int,
    requested: int | None = None,
    *,
    agents_per_sector: int = cfg.AGENTS_PER_SECTOR,
    max_sectors: int | None = None,
) -> int:
    """How many sectors - and therefore parallel tasks - to use this round.

    Depends only on the population, never on the worker count: two machines
    with a different number of cores must simulate the same run. ``max_sectors``
    is the run's ceiling, derived from the *starting* population; without it the
    ceiling is derived from this round's population.
    """
    total = int(agent_count)
    if total <= 0:
        return 0

    cap = max_sectors_for(total) if max_sectors is None else max(1, int(max_sectors))

    if requested is None:
        requested = cfg.SECTOR_COUNT
    try:
        requested = int(requested or 0)
    except (TypeError, ValueError):
        requested = 0

    if requested > 0:
        # An explicit request wins (that is how a run is reproduced exactly on
        # another machine), but it is still bounded by the run's ceiling.
        return max(1, min(requested, cap, total))

    per_sector = max(1, int(agents_per_sector))
    return max(1, min(math.ceil(total / per_sector), cap, total))


def build_sectors(
    positions: dict[int, Point],
    target_k: int,
    padding: float = cfg.SECTOR_PADDING,
    types: dict[int, str] | None = None,
    map_bounds: Bounds | None = None,
) -> list[Sector]:
    """Group agents into sectors with a grid cut of the map.

    ``target_k`` is authoritative (the caller has already applied the
    population policy via :func:`resolve_sector_count`).

    ``types`` maps agent id to ``"human"``/``"zombie"``. When given, cells that
    hold only one type are merged into the nearest mixed cell: a sector without
    both types can never produce a fight, and leaving one idle is what let runs
    finish with both sides alive.
    """
    if not positions:
        return []

    ids = list(positions.keys())
    total = len(ids)
    bounds = map_bounds if map_bounds is not None else map_bounds_for(total)

    # The caller already applied the population policy (see
    # ``resolve_sector_count``); the target is authoritative.
    k = max(1, min(int(target_k), total))

    labels = _grid_labels(ids, positions, bounds, k)

    groups: dict[int, list[int]] = {}
    for agent_id, label in zip(ids, labels):
        groups.setdefault(label, []).append(agent_id)

    clusters = [members for members in groups.values() if members]
    if types:
        clusters = _merge_mono_type(clusters, positions, types)
    clusters.sort(key=lambda members: _sort_key(members, positions))

    return [
        _make_sector(index, members, positions, padding, bounds)
        for index, members in enumerate(clusters, start=1)
    ]


def _grid_labels(
    ids: list[int],
    positions: dict[int, Point],
    bounds: Bounds,
    k: int,
) -> list[int]:
    """Return the cell label for each agent in a ``k``-cell grid cut.

    ``floor(sqrt(k))`` columns keeps the cell count at or below the target: a
    5-cell target becomes 2x2, a 10-cell target becomes 3x3. Points outside the
    map (only the tests do this) are clamped to the nearest edge cell.
    """
    min_x, min_y, max_x, max_y = bounds
    cols = max(1, int(math.sqrt(k)))
    rows = max(1, k // cols)
    width = max_x - min_x
    height = max_y - min_y

    labels: list[int] = []
    for agent_id in ids:
        x, y = positions[agent_id]
        col = int((x - min_x) / width * cols) if width > 0 else 0
        row = int((y - min_y) / height * rows) if height > 0 else 0
        col = min(cols - 1, max(0, col))
        row = min(rows - 1, max(0, row))
        labels.append(row * cols + col)
    return labels


def _sort_key(members: list[int], positions: dict[int, Point]):
    """Stable ordering: biggest cluster first, then by left-most agent."""
    xs = [positions[agent_id][0] for agent_id in members]
    ys = [positions[agent_id][1] for agent_id in members]
    return (-len(members), min(xs), min(ys))


def _make_sector(
    index: int,
    members: list[int],
    positions: dict[int, Point],
    padding: float,
    map_bounds: Bounds,
) -> Sector:
    min_x, min_y, max_x, max_y = map_bounds
    xs = [positions[agent_id][0] for agent_id in members]
    ys = [positions[agent_id][1] for agent_id in members]
    bounds = (
        max(min_x, min(xs) - padding),
        max(min_y, min(ys) - padding),
        min(max_x, max(xs) + padding),
        min(max_y, max(ys) + padding),
    )
    return Sector(index=index, agent_ids=members, bounds=bounds)


def _merge_mono_type(
    clusters: list[list[int]],
    positions: dict[int, Point],
    types: dict[int, str],
) -> list[list[int]]:
    """Absorb single-type clusters into the nearest mixed one.

    A sector with only humans (or only zombies) has nobody to fight, so it is
    pure overhead and, worse, it lets a run stall with both sides alive. When
    nothing is mixed the whole map becomes a single arena - that is the only
    arrangement that guarantees the two sides can reach each other.
    """
    mixed: list[list[int]] = []
    mono: list[list[int]] = []
    for members in clusters:
        kinds = {types.get(agent_id) for agent_id in members}
        (mixed if len(kinds) > 1 else mono).append(members)

    if not mixed:
        return [[agent_id for members in clusters for agent_id in members]]
    if not mono:
        return mixed

    counts = [len(members) for members in mixed]
    sums = [_sum_positions(members, positions) for members in mixed]
    for members in mono:
        count = len(members)
        sum_x, sum_y = _sum_positions(members, positions)
        center = (sum_x / count, sum_y / count)
        best = min(
            range(len(mixed)),
            key=lambda i: _dist2(
                center, (sums[i][0] / counts[i], sums[i][1] / counts[i])
            ),
        )
        mixed[best].extend(members)
        counts[best] += count
        sums[best] = (sums[best][0] + sum_x, sums[best][1] + sum_y)
    return mixed


def _sum_positions(members: list[int], positions: dict[int, Point]) -> Point:
    """Sum of x and y over one cluster in a single pass."""
    sum_x = 0.0
    sum_y = 0.0
    for agent_id in members:
        x, y = positions[agent_id]
        sum_x += x
        sum_y += y
    return sum_x, sum_y


def _dist2(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy
