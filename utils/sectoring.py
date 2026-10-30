"""Cluster-based sectoring.

Agents are grouped with a small k-means implementation (no third-party
dependency). Each cluster becomes one sector with its own walkable bounding
box, so only a subset of the world is ever resident in memory.

The number of clusters is driven by the population alone:

    k = clamp(ceil(agents / AGENTS_PER_SECTOR), 1, MAX_SECTORS)

Fewer, larger sectors are deliberate. Sectors are the unit of parallel work,
but they also decide who can ever meet: a finely split map leaves sectors
containing only humans or only zombies, those sectors cannot do anything, and
the run ends with both sides alive. A coarse split keeps the two sides in the
same arena, which is what lets a run reach a real conclusion.

Every sector is also guaranteed to hold both types (mono-type clusters are
merged into the nearest mixed one), so no parallel work is ever spent on a
sector that cannot fight.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from config import simulation_config as cfg

Point = tuple[float, float]
_INF = float("inf")


@dataclass
class Sector:
    index: int
    agent_ids: list[int] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (
        float(cfg.MAP_MIN),
        float(cfg.MAP_MIN),
        float(cfg.MAP_MAX),
        float(cfg.MAP_MAX),
    )

    @property
    def size(self) -> int:
        return len(self.agent_ids)

    def describe(self) -> str:
        left, bottom, right, top = self.bounds
        return (
            f"sector {self.index}: {self.size} agents, "
            f"x[{left:.0f}, {right:.0f}] y[{bottom:.0f}, {top:.0f}]"
        )


def resolve_sector_count(
    agent_count: int,
    requested: int | None = None,
    *,
    agents_per_sector: int = cfg.AGENTS_PER_SECTOR,
    max_sectors: int = cfg.MAX_SECTORS,
) -> int:
    """How many sectors - and therefore parallel tasks - to use this round.

    Depends only on the population, never on the worker count: two machines
    with a different number of cores must simulate the same run.
    """
    total = int(agent_count)
    if total <= 0:
        return 0

    if requested is None:
        requested = cfg.SECTOR_COUNT
    try:
        requested = int(requested or 0)
    except (TypeError, ValueError):
        requested = 0

    if requested > 0:
        # An explicit request wins (that is how a run is reproduced exactly on
        # another machine), but it is still bounded by the schema limits.
        return max(1, min(requested, int(max_sectors), total))

    per_sector = max(1, int(agents_per_sector))
    return max(1, min(math.ceil(total / per_sector), int(max_sectors), total))


def build_sectors(
    positions: dict[int, Point],
    rng: random.Random | None = None,
    max_sectors: int = cfg.MAX_SECTORS,
    padding: float = cfg.SECTOR_PADDING,
    target_k: int | None = None,
    types: dict[int, str] | None = None,
) -> list[Sector]:
    """Group agents into clustered sectors based on their positions.

    ``target_k`` is authoritative when given (the caller has already applied the
    population policy). Without it the same population rule is used.

    ``types`` maps agent id to ``"human"``/``"zombie"``. When given, clusters
    that hold only one type are merged into the nearest mixed cluster: a sector
    without both types can never produce a fight, and leaving one idle is what
    let runs finish with both sides alive.
    """
    if not positions:
        return []

    rng = rng or random.Random(0)
    ids = list(positions.keys())
    total = len(ids)

    if target_k is None:
        k = resolve_sector_count(total, max_sectors=max_sectors)
    else:
        k = max(1, min(int(target_k), max_sectors, total))

    points = [positions[agent_id] for agent_id in ids]
    labels = _kmeans(points, k, rng) if k > 1 else [0] * total

    groups: dict[int, list[int]] = {}
    for agent_id, label in zip(ids, labels):
        groups.setdefault(label, []).append(agent_id)

    clusters = [members for members in groups.values() if members]
    if types:
        clusters = _merge_mono_type(clusters, positions, types)
    clusters.sort(key=lambda members: _sort_key(members, positions))

    return [
        _make_sector(index, members, positions, padding)
        for index, members in enumerate(clusters, start=1)
    ]


def _sort_key(members: list[int], positions: dict[int, Point]):
    """Stable ordering: biggest cluster first, then by left-most agent."""
    xs = [positions[agent_id][0] for agent_id in members]
    ys = [positions[agent_id][1] for agent_id in members]
    return (-len(members), min(xs), min(ys))


def _make_sector(
    index: int, members: list[int], positions: dict[int, Point], padding: float
) -> Sector:
    xs = [positions[agent_id][0] for agent_id in members]
    ys = [positions[agent_id][1] for agent_id in members]
    bounds = (
        max(cfg.MAP_MIN, min(xs) - padding),
        max(cfg.MAP_MIN, min(ys) - padding),
        min(cfg.MAP_MAX, max(xs) + padding),
        min(cfg.MAP_MAX, max(ys) + padding),
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

    centers = [_centroid(members, positions) for members in mixed]
    for members in mono:
        center = _centroid(members, positions)
        best = min(range(len(mixed)), key=lambda i: _dist2(center, centers[i]))
        mixed[best].extend(members)
        # Refresh the centre so later merges still pick the closest cluster.
        centers[best] = _centroid(mixed[best], positions)
    return mixed


def _centroid(members: list[int], positions: dict[int, Point]) -> Point:
    count = len(members)
    return (
        sum(positions[agent_id][0] for agent_id in members) / count,
        sum(positions[agent_id][1] for agent_id in members) / count,
    )


def _kmeans(points: list[Point], k: int, rng: random.Random, max_iter: int = 40):
    """Tiny k-means with k-means++ seeding. Returns a label per point.

    The inner loops are written with plain local variables instead of
    ``min(range(...), key=lambda ...)``; the result is identical (ties go to the
    lowest index) but several times faster, which matters because clustering
    runs serially between rounds.
    """
    distinct = list({point for point in points})
    if len(distinct) <= k:
        lookup = {point: index for index, point in enumerate(distinct)}
        return [lookup[point] for point in points]

    centers: list[Point] = [points[rng.randrange(len(points))]]
    while len(centers) < k:
        weights = [
            min(_dist2(point, center) for center in centers) for point in points
        ]
        total = sum(weights)
        if total <= 0:
            break
        threshold = rng.random() * total
        running = 0.0
        for point, weight in zip(points, weights):
            running += weight
            if running >= threshold:
                centers.append(point)
                break

    labels = [-1] * len(points)
    count = len(points)
    for _ in range(max_iter):
        changed = False
        for i in range(count):
            px, py = points[i]
            best = 0
            best_distance = _INF
            for c in range(len(centers)):
                cx, cy = centers[c]
                dx = px - cx
                dy = py - cy
                distance = dx * dx + dy * dy
                if distance < best_distance:
                    best_distance = distance
                    best = c
            if labels[i] != best:
                labels[i] = best
                changed = True

        sums = [[0.0, 0.0, 0] for _ in centers]
        for label, (px, py) in zip(labels, points):
            bucket = sums[label]
            bucket[0] += px
            bucket[1] += py
            bucket[2] += 1
        centers = [
            (sx / members, sy / members) if members else centers[i]
            for i, (sx, sy, members) in enumerate(sums)
        ]
        if not changed:
            break

    return labels


def _dist2(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy
