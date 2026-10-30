"""One sector's simulation, with no database and no visualisers.

Everything a sector visit does lives here, so the single-threaded path and the
worker paths run *exactly* the same code.  The engine only mutates the agent
objects it is handed and returns a report plus (optionally) the frames that
were generated, which the caller replays on the main thread.

The randomness is seeded per sector visit via :func:`derive_seed`.  That is
what makes a run reproducible regardless of how many workers execute it: a
sector always sees the same random stream whether it ran first, last, alone or
in a pool.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Mapping

from config import simulation_config as cfg
from src.agent import Agent

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


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class SectorEngine:
    def __init__(self, max_ticks: int | None = None, capture_frames: bool = False):
        if max_ticks is None:
            max_ticks = max(
                1, min(cfg.SECTOR_REFRESH_TICKS, cfg.MAX_ITERATIONS_PER_SECTOR)
            )
        self.max_ticks = int(max_ticks)
        self.capture_frames = bool(capture_frames)

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

        while iteration < self.max_ticks:
            iteration += 1

            moved, gap = self._movement_phase(sector, humans, zombies)
            moved_total += moved
            events, attacks = self._combat_phase(humans, zombies, rng)

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
        sector, humans: Mapping[int, Agent], zombies: Mapping[int, Agent]
    ) -> tuple[float, float]:
        """Every living agent walks toward its nearest living enemy.

        Humans walk only until their target is inside weapon range, then hold
        and fire from there. Zombies keep closing to ``ZOMBIE_STOP_DISTANCE``.

        Returns ``(total distance moved, distance to the closest enemy pair)``.
        The second value feeds the stall detector, so a sector that cannot make
        progress is stopped even if combat never started.
        """
        moved_total = 0.0
        closest_gap = float("inf")
        bounds = sector.bounds

        for human in humans.values():
            if not human.is_alive:
                continue
            target, distance = SectorEngine._nearest(human, zombies.values())
            if target is None:
                continue
            closest_gap = min(closest_gap, distance)
            moved_total += human.move_toward(
                target.x_pos,
                target.y_pos,
                stop_distance=min(human.hold_distance, distance),
                bounds=bounds,
            )

        for zombie in zombies.values():
            if not zombie.is_alive:
                continue
            target, distance = SectorEngine._nearest(zombie, humans.values())
            if target is None:
                continue
            closest_gap = min(closest_gap, distance)
            moved_total += zombie.move_toward(
                target.x_pos,
                target.y_pos,
                stop_distance=min(cfg.ZOMBIE_STOP_DISTANCE, distance),
                bounds=bounds,
            )

        return moved_total, closest_gap

    @staticmethod
    def _combat_phase(
        humans: Mapping[int, Agent],
        zombies: Mapping[int, Agent],
        rng: random.Random,
    ) -> tuple[list[str], int]:
        """Resolve every attack from the same snapshot (simultaneous)."""
        events: list[str] = []
        damage: dict[int, int] = {}
        attacks = 0

        for human in humans.values():
            if not human.is_alive:
                continue
            target, distance = SectorEngine._nearest(human, zombies.values())
            if target is None or not human.can_reach(distance):
                continue

            attacks += 1
            if rng.random() <= human.hit_chance(distance):
                dealt = human.roll_damage(rng)
                damage[target.agent_id] = damage.get(target.agent_id, 0) + dealt
                events.append(
                    f"H{human.agent_id} hit Z{target.agent_id} "
                    f"for {dealt} (tier {human.weapon_tier}, {distance:.1f}u)"
                )
            else:
                events.append(
                    f"H{human.agent_id} missed Z{target.agent_id} "
                    f"(tier {human.weapon_tier}, {distance:.1f}u)"
                )

        for zombie in zombies.values():
            if not zombie.is_alive:
                continue
            target, distance = SectorEngine._nearest(zombie, humans.values())
            if target is None or not zombie.can_reach(distance):
                continue

            attacks += 1
            dealt = zombie.roll_damage(rng)
            damage[target.agent_id] = damage.get(target.agent_id, 0) + dealt
            events.append(
                f"Z{zombie.agent_id} hit H{target.agent_id} "
                f"for {dealt} (strength {zombie.strength}, {distance:.1f}u)"
            )

        # Apply simultaneously - nobody dies before swinging this iteration.
        if damage:
            by_id = {
                agent.agent_id: agent
                for agent in (*humans.values(), *zombies.values())
            }
            for agent_id, total in damage.items():
                by_id[agent_id].take_damage(total)

        return events, attacks

    @staticmethod
    def _nearest(agent: Agent, candidates) -> tuple[Agent | None, float]:
        """Closest living candidate (squared distances inside the loop).

        The inner loop is the hot path of the whole simulation - every agent
        scans every enemy, every tick - so it compares squared distances with
        plain locals instead of calling ``distance_to`` (and taking a square
        root) per candidate. The winner, and therefore the behaviour, is the
        same.
        """
        best: Agent | None = None
        best_sq = float("inf")
        ax, ay = agent.x_pos, agent.y_pos
        for candidate in candidates:
            if not candidate.is_alive:
                continue
            dx = candidate.x_pos - ax
            dy = candidate.y_pos - ay
            dist_sq = dx * dx + dy * dy
            if dist_sq < best_sq:
                best, best_sq = candidate, dist_sq
        if best is None:
            return None, float("inf")
        return best, math.sqrt(best_sq)

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
            "event_count": len(events),
        }
        frames.append((snapshot, meta))
