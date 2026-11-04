"""One sector's simulation, with no database and no visualisers.

Everything a sector visit does lives here, so the single-threaded path and the
worker paths run *exactly* the same code.  The engine only mutates the agent
objects it is handed and returns a report plus (optionally) the frames that
were generated, which the caller replays on the main thread.

The randomness is seeded per sector visit via :func:`derive_seed`.  That is
what makes a run reproducible regardless of how many workers execute it: a
sector always sees the same random stream whether it ran first, last, alone or
in a pool.

Parallelism inside a sector is deliberately limited to *planning*: workers
compute movement deltas and attack targets from a read-only snapshot, and the
engine applies them in canonical order.  RNG draws and damage resolution stay
on one thread, so chunking cannot change a seeded run.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from functools import partial
from typing import Mapping

from config import simulation_config as cfg
from src.agent import Agent
from src.tick_executor import chunked, make_executor
from utils.spatial import SpatialIndex

_U64 = 0xFFFF_FFFF_FFFF_FFFF


def derive_seed(round_seed: int, sector_index: int, salt: int = 0) -> int:
    """Stable 64-bit seed for one sector visit.

    ``hashlib`` is used instead of :func:`hash` because the built-in hash of a
    tuple is salted per process and would break reproducibility across worker
    processes.
    """
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(round_seed & _U64).to_bytes(8, "big"))
    digest.update(int(sector_index & _U64).to_bytes(8, "big"))
    digest.update(int(salt & _U64).to_bytes(8, "big"))
    return int.from_bytes(digest.digest(), "big")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class SectorReport:
    index: int
    bounds: tuple[float, float, float, float]
    humans_start: int
    zombies_start: int
    humans_end: int
    zombies_end: int
    iterations: int
    attacks: int
    stop_reason: str
    round_index: int = 1


@dataclass
class SectorOutcome:
    """What a sector visit leaves behind for the orchestrator."""

    report: SectorReport
    moved: float = 0.0
    frames: list = field(default_factory=list)
    # (agent_id, agent_type, x, y, is_alive) for every agent the visit owned.
    # The orchestrator can cluster the next round from these instead of doing
    # another full SELECT, as long as the values match what was written to the
    # database (see ``Simulation._next_rows``).
    final_states: list = field(default_factory=list)


@dataclass(frozen=True)
class MovePlan:
    """A movement computed from a snapshot, not yet applied."""

    agent_id: int
    new_x: float
    new_y: float
    moved: float
    gap: float


@dataclass(frozen=True)
class AttackPlan:
    """The nearest enemy of one attacker, computed from a snapshot."""

    attacker_id: int
    target_id: int
    distance: float


def _flatten(chunks):
    for chunk in chunks:
        yield from chunk


def shard_parts(count: int, shards: int, chunk_size: int = 0) -> int:
    """How many chunks to split ``count`` agents into.

    ``chunk_size`` is a load-balancing cap: it can raise the chunk count but
    never lowers it below the requested shard count.
    """
    parts = max(1, int(shards or 1))
    if chunk_size:
        parts = max(parts, math.ceil(count / chunk_size))
    return max(1, min(parts, max(1, count)))


# --------------------------------------------------------------------------- #
# Plan functions (safe to run on a worker thread)
# --------------------------------------------------------------------------- #
def _plan_human_moves(agents, index: SpatialIndex, bounds):
    plans = []
    for human in agents:
        if not human.is_alive:
            continue
        target, distance = index.nearest(human.x_pos, human.y_pos)
        if target is None:
            continue
        new_x, new_y, moved = human.planned_move(
            target.x_pos,
            target.y_pos,
            stop_distance=min(human.hold_distance, distance),
            bounds=bounds,
        )
        plans.append(MovePlan(human.agent_id, new_x, new_y, moved, distance))
    return plans


def _plan_zombie_moves(agents, index: SpatialIndex, bounds):
    plans = []
    for zombie in agents:
        if not zombie.is_alive:
            continue
        target, distance = index.nearest(zombie.x_pos, zombie.y_pos)
        if target is None:
            continue
        new_x, new_y, moved = zombie.planned_move(
            target.x_pos,
            target.y_pos,
            stop_distance=min(cfg.ZOMBIE_STOP_DISTANCE, distance),
            bounds=bounds,
        )
        plans.append(MovePlan(zombie.agent_id, new_x, new_y, moved, distance))
    return plans


def _plan_attacks(agents, index: SpatialIndex):
    plans = []
    for agent in agents:
        if not agent.is_alive:
            continue
        target, distance = index.nearest(agent.x_pos, agent.y_pos)
        if target is None:
            continue
        plans.append(AttackPlan(agent.agent_id, target.agent_id, distance))
    return plans


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class SectorEngine:
    def __init__(
        self,
        max_ticks: int | None = None,
        capture_frames: bool = False,
        event_shards: int = 1,
        chunk_size: int | None = None,
        frame_interval: int | None = None,
    ):
        if max_ticks is None:
            max_ticks = max(
                1, min(cfg.SECTOR_REFRESH_TICKS, cfg.MAX_ITERATIONS_PER_SECTOR)
            )
        self.max_ticks = int(max_ticks)
        self.capture_frames = bool(capture_frames)
        self.event_shards = max(1, int(event_shards or 1))
        # 0/None = split into at most ``event_shards`` even chunks.
        raw_chunk = cfg.EVENT_CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_size = max(0, int(raw_chunk or 0))
        raw_interval = cfg.FRAME_INTERVAL if frame_interval is None else frame_interval
        self.frame_interval = max(1, int(raw_interval or 1))

    # ------------------------------------------------------------------ #
    def run(
        self,
        sector,
        humans: Mapping[int, Agent],
        zombies: Mapping[int, Agent],
        rng: random.Random,
        round_index: int = 1,
        sectors_in_round: int = 0,
    ) -> SectorOutcome:
        humans_start = len(humans)
        zombies_start = len(zombies)

        report = SectorReport(
            index=sector.index,
            bounds=sector.bounds,
            humans_start=humans_start,
            zombies_start=zombies_start,
            humans_end=humans_start,
            zombies_end=zombies_start,
            iterations=0,
            attacks=0,
            stop_reason="no_enemies",
            round_index=round_index,
        )
        frames: list = []
        context = (sector, humans, zombies, round_index, sectors_in_round, frames)

        if not humans_start or not zombies_start:
            self._emit(
                *context,
                0,
                [],
                False,
                ended=True,
                note=(
                    "sector contains only humans"
                    if humans_start
                    else "sector contains only zombies"
                    if zombies_start
                    else "sector empty"
                ),
            )
            return SectorOutcome(report=report, moved=0.0, frames=frames)

        # A visit is deliberately short: the map is re-clustered after this
        # many ticks, so no single sector can grind on for thousands of ticks.
        combat_started = False
        quiet_iterations = 0
        stall_iterations = 0
        best_gap = float("inf")
        moved_total = 0.0
        iteration = 0
        report.stop_reason = "tick budget"

        executor = make_executor(self.event_shards)
        try:
            while iteration < self.max_ticks:
                iteration += 1

                moved, gap, human_index = self._movement_phase(
                    sector,
                    humans,
                    zombies,
                    executor,
                    self.event_shards,
                    self.chunk_size,
                )
                moved_total += moved
                events, attacks = self._combat_phase(
                    sector,
                    humans,
                    zombies,
                    rng,
                    executor,
                    self.event_shards,
                    self.chunk_size,
                    collect_events=self.capture_frames,
                    human_index=human_index,
                )

                alive_h = sum(1 for agent in humans.values() if agent.is_alive)
                alive_z = sum(1 for agent in zombies.values() if agent.is_alive)

                report.attacks += attacks
                if attacks:
                    combat_started = True
                    quiet_iterations = 0
                    stall_iterations = 0
                    best_gap = float("inf")
                else:
                    quiet_iterations += 1
                    # Has the closest human/zombie pair got any closer?
                    if gap < best_gap - 1e-9:
                        best_gap = gap
                        stall_iterations = 0
                    else:
                        stall_iterations += 1

                self._emit(*context, iteration, events, attacks > 0)

                if alive_h == 0 or alive_z == 0:
                    report.stop_reason = "one side eliminated"
                    break
                if attacks == 0 and moved < 1e-6:
                    report.stop_reason = "no combat, no movement"
                    break
                if attacks == 0 and stall_iterations >= cfg.STALL_PATIENCE:
                    report.stop_reason = "no progress"
                    break
                if combat_started and quiet_iterations >= cfg.NO_COMBAT_PATIENCE:
                    report.stop_reason = "no combat"
                    break
        finally:
            executor.close()

        report.humans_end = sum(1 for agent in humans.values() if agent.is_alive)
        report.zombies_end = sum(1 for agent in zombies.values() if agent.is_alive)
        report.iterations = iteration

        self._emit(*context, iteration, [], False, ended=True, note=report.stop_reason)
        return SectorOutcome(report=report, moved=moved_total, frames=frames)

    # ------------------------------------------------------------------ #
    # Phases
    # ------------------------------------------------------------------ #
    @staticmethod
    def _movement_phase(
        sector,
        humans: Mapping[int, Agent],
        zombies: Mapping[int, Agent],
        executor=None,
        shards: int = 1,
        chunk_size: int = 0,
    ) -> tuple[float, float, SpatialIndex]:
        """Every living agent walks toward its nearest living enemy.

        Humans move first against the pre-move zombie snapshot; zombies then
        move against the post-human snapshot.  That is the same order the
        serial loop always used, and it lets each group run in parallel because
        nobody in a group reads another mover's new position.

        Returns ``(total distance moved, distance to the closest enemy pair,
        post-move human index)``.  The gap feeds the stall detector, and the
        index is reused by the combat phase because humans do not move again.
        """
        owned = executor is None
        if owned:
            executor = make_executor(shards)
        try:
            humans_list = list(humans.values())
            zombies_list = list(zombies.values())
            moved_total = 0.0
            closest_gap = float("inf")
            bounds = getattr(sector, "bounds", None)

            # Humans: read-only zombie positions from before this tick's moves.
            zombie_index = SpatialIndex(zombies_list, bounds=bounds)
            human_planner = partial(
                _plan_human_moves, index=zombie_index, bounds=bounds
            )
            for plans in executor.map(
                human_planner,
                chunked(humans_list, shard_parts(len(humans_list), shards, chunk_size)),
            ):
                for plan in plans:
                    human = humans[plan.agent_id]
                    human.x_pos = plan.new_x
                    human.y_pos = plan.new_y
                    moved_total += plan.moved
                    if plan.gap < closest_gap:
                        closest_gap = plan.gap

            # Zombies: humans have moved, so rebuild before planning their chase.
            human_index = SpatialIndex(humans_list, bounds=bounds)
            zombie_planner = partial(
                _plan_zombie_moves, index=human_index, bounds=bounds
            )
            for plans in executor.map(
                zombie_planner,
                chunked(zombies_list, shard_parts(len(zombies_list), shards, chunk_size)),
            ):
                for plan in plans:
                    zombie = zombies[plan.agent_id]
                    zombie.x_pos = plan.new_x
                    zombie.y_pos = plan.new_y
                    moved_total += plan.moved
                    if plan.gap < closest_gap:
                        closest_gap = plan.gap

            return moved_total, closest_gap, human_index
        finally:
            if owned:
                executor.close()

    @staticmethod
    def _combat_phase(
        sector,
        humans: Mapping[int, Agent],
        zombies: Mapping[int, Agent],
        rng: random.Random,
        executor=None,
        shards: int = 1,
        chunk_size: int = 0,
        collect_events: bool = True,
        human_index: SpatialIndex | None = None,
    ) -> tuple[list[str], int]:
        """Resolve every attack from the same snapshot (simultaneous).

        Targeting is planned in parallel; RNG rolls and damage application are
        merged in canonical order (humans, then zombies, ascending id) so the
        random stream is identical to the serial loop.  ``human_index`` is the
        post-move index from the movement phase; pass it to avoid rebuilding it.
        """
        owned = executor is None
        if owned:
            executor = make_executor(shards)
        try:
            bounds = getattr(sector, "bounds", None)
            humans_list = list(humans.values())
            zombies_list = list(zombies.values())

            if human_index is None:
                human_index = SpatialIndex(humans_list, bounds=bounds)
            zombie_index = SpatialIndex(zombies_list, bounds=bounds)

            human_plan_chunks = executor.map(
                partial(_plan_attacks, index=zombie_index),
                chunked(
                    humans_list, shard_parts(len(humans_list), shards, chunk_size)
                ),
            )
            zombie_plan_chunks = executor.map(
                partial(_plan_attacks, index=human_index),
                chunked(
                    zombies_list, shard_parts(len(zombies_list), shards, chunk_size)
                ),
            )
            human_plans = _flatten(human_plan_chunks)
            zombie_plans = _flatten(zombie_plan_chunks)

            events: list[str] = []
            damage: dict[int, int] = {}
            attacks = 0
            human_by_id = {
                agent.agent_id: agent for agent in humans_list if agent.is_alive
            }
            zombie_by_id = {
                agent.agent_id: agent for agent in zombies_list if agent.is_alive
            }

            for plan in human_plans:
                human = human_by_id.get(plan.attacker_id)
                target = zombie_by_id.get(plan.target_id)
                if human is None or target is None or not human.can_reach(
                    plan.distance
                ):
                    continue

                attacks += 1
                if rng.random() <= human.hit_chance(plan.distance):
                    dealt = human.roll_damage(rng)
                    damage[target.agent_id] = damage.get(target.agent_id, 0) + dealt
                    if collect_events:
                        events.append(
                            f"H{human.agent_id} hit Z{target.agent_id} "
                            f"for {dealt} (tier {human.weapon_tier}, "
                            f"{plan.distance:.1f}u)"
                        )
                elif collect_events:
                    events.append(
                        f"H{human.agent_id} missed Z{target.agent_id} "
                        f"(tier {human.weapon_tier}, {plan.distance:.1f}u)"
                    )

            for plan in zombie_plans:
                zombie = zombie_by_id.get(plan.attacker_id)
                target = human_by_id.get(plan.target_id)
                if zombie is None or target is None or not zombie.can_reach(
                    plan.distance
                ):
                    continue

                attacks += 1
                dealt = zombie.roll_damage(rng)
                damage[target.agent_id] = damage.get(target.agent_id, 0) + dealt
                if collect_events:
                    events.append(
                        f"Z{zombie.agent_id} hit H{target.agent_id} "
                        f"for {dealt} (strength {zombie.strength}, "
                        f"{plan.distance:.1f}u)"
                    )

            # Apply simultaneously - nobody dies before swinging this iteration.
            if damage:
                by_id = {
                    agent.agent_id: agent
                    for agent in (*humans_list, *zombies_list)
                }
                for agent_id, total in damage.items():
                    by_id[agent_id].take_damage(total)

            return events, attacks
        finally:
            if owned:
                executor.close()

    # ------------------------------------------------------------------ #
    # Frames
    # ------------------------------------------------------------------ #
    def _emit(
        self,
        sector,
        humans: Mapping[int, Agent],
        zombies: Mapping[int, Agent],
        round_index: int,
        sectors_in_round: int,
        frames: list,
        iteration: int,
        events: list[str],
        combat: bool,
        ended: bool = False,
        note: str = "",
    ) -> None:
        if not self.capture_frames:
            return
        # Every Nth tick, plus the first frame and the closing one, keeps the
        # HTML payload bounded without losing the start/end state.
        if (
            self.frame_interval > 1
            and not ended
            and iteration > 1
            and iteration % self.frame_interval
        ):
            return

        snapshot = [
            agent.snapshot() for agent in (*humans.values(), *zombies.values())
        ]
        meta = {
            "sector": sector.index,
            "sectors": sectors_in_round,
            "round": round_index,
            "sector_size": sector.size,
            "iteration": iteration,
            "humans_alive": sum(1 for agent in humans.values() if agent.is_alive),
            "zombies_alive": sum(1 for agent in zombies.values() if agent.is_alive),
            "combat": combat,
            "ended": ended,
            "note": note,
            "bounds": [float(value) for value in sector.bounds],
            "events": events[:10],
        }
        frames.append((snapshot, meta))
