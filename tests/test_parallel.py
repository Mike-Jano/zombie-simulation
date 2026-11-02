"""Tests for parallel sectoring: policy, determinism and backend equivalence."""

from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.capabilities import resolve_plan  # noqa: E402
from runtime.sector_pool import SectorPool  # noqa: E402
from src.sector_engine import derive_seed  # noqa: E402
from src.sector_task import SectorTask, simulate_sector_task  # noqa: E402
from src.simulation import Simulation  # noqa: E402
from src.spawner import build_agents  # noqa: E402
from utils.sectoring import build_sectors, resolve_sector_count  # noqa: E402

SEED = 4242


def make_agents(humans: int, zombies: int, seed: int = SEED) -> list:
    return build_agents(humans, zombies, random.Random(seed))


def make_tasks(agents, sector_count: int, seed: int = SEED):
    positions = {
        agent.agent_id: (float(agent.x_pos), float(agent.y_pos)) for agent in agents
    }
    types = {agent.agent_id: agent.agent_type for agent in agents}
    sector_list = build_sectors(
        positions, rng=random.Random(seed), target_k=sector_count, types=types
    )
    by_id = {agent.agent_id: agent for agent in agents}
    tasks = []
    for sector in sector_list:
        members = [by_id[agent_id] for agent_id in sector.agent_ids]
        tasks.append(
            SectorTask(
                sector=sector,
                seed=derive_seed(seed, sector.index),
                round_index=1,
                sectors_in_round=len(sector_list),
                humans=[a for a in members if a.agent_type == "human"],
                zombies=[a for a in members if a.agent_type == "zombie"],
            )
        )
    return tasks, sector_list


def state_of(tasks) -> dict:
    """Round-trip-safe snapshot of every agent a task list owns."""
    state = {}
    for task in tasks:
        for agent in [*(task.humans or []), *(task.zombies or [])]:
            state[agent.agent_id] = (
                round(agent.x_pos, 6),
                round(agent.y_pos, 6),
                int(agent.health),
                bool(agent.is_alive),
            )
    return state


class SectorPolicyTests(unittest.TestCase):
    def test_sectors_scale_with_population(self):
        # 500 agents per sector, capped at MAX_SECTORS.
        self.assertEqual(resolve_sector_count(200), 1)
        self.assertEqual(resolve_sector_count(500), 1)
        self.assertEqual(resolve_sector_count(501), 2)
        self.assertEqual(resolve_sector_count(1500), 3)
        self.assertEqual(resolve_sector_count(3000), 6)
        self.assertEqual(resolve_sector_count(6000), 12)
        self.assertEqual(resolve_sector_count(60_000), 12)

    def test_worker_count_does_not_change_the_plan(self):
        # There is no worker argument any more: cores cannot reshape the run.
        self.assertEqual(resolve_sector_count(200), 1)
        self.assertEqual(resolve_sector_count(10), 1)
        self.assertEqual(resolve_sector_count(0), 0)

    def test_explicit_request_wins(self):
        self.assertEqual(resolve_sector_count(1200, requested=3), 3)
        # ...but never beyond the schema/sector limits.
        self.assertEqual(resolve_sector_count(1200, requested=99), 12)

    def test_build_sectors_honours_target(self):
        agents = make_agents(200, 200)
        positions = {a.agent_id: (a.x_pos, a.y_pos) for a in agents}
        types = {a.agent_id: a.agent_type for a in agents}
        sectors = build_sectors(
            positions, rng=random.Random(1), target_k=6, types=types
        )
        self.assertEqual(len(sectors), 6)
        self.assertEqual(
            sum(len(sector.agent_ids) for sector in sectors), len(agents)
        )
        # Every agent belongs to exactly one sector.
        owners = [agent_id for sector in sectors for agent_id in sector.agent_ids]
        self.assertEqual(len(owners), len(set(owners)))

    def test_every_sector_holds_both_types(self):
        agents = make_agents(300, 300)
        positions = {a.agent_id: (a.x_pos, a.y_pos) for a in agents}
        types = {a.agent_id: a.agent_type for a in agents}
        sectors = build_sectors(
            positions, rng=random.Random(3), target_k=10, types=types
        )
        for sector in sectors:
            kinds = {types[agent_id] for agent_id in sector.agent_ids}
            self.assertEqual(kinds, {"human", "zombie"}, sector.describe())

    def test_segregated_map_collapses_to_one_arena(self):
        # Humans on the left, zombies on the right: every spatial cluster is
        # single-type, so sectoring must fall back to one arena.
        positions = {}
        types = {}
        for agent_id in range(1, 101):
            positions[agent_id] = (-400.0 + agent_id, 0.0)
            types[agent_id] = "human"
        for agent_id in range(101, 201):
            positions[agent_id] = (300.0 + agent_id, 0.0)
            types[agent_id] = "zombie"

        sectors = build_sectors(
            positions, rng=random.Random(0), target_k=4, types=types
        )
        self.assertEqual(len(sectors), 1)
        self.assertEqual(len(sectors[0].agent_ids), 200)

    def test_mono_type_cluster_merges_into_a_mixed_one(self):
        # Two mixed clusters plus a lone zombie far from everything: it must
        # join a fight instead of sitting in its own idle sector.
        positions = {}
        types = {}
        for offset, base in enumerate((0.0, 200.0)):
            for i in range(4):
                agent_id = offset * 4 + i + 1
                positions[agent_id] = (base + i, base + (i % 2))
                types[agent_id] = "human" if i % 2 == 0 else "zombie"
        positions[99] = (-400.0, 300.0)
        types[99] = "zombie"

        sectors = build_sectors(
            positions, rng=random.Random(0), target_k=3, types=types
        )
        self.assertEqual(
            sorted(agent_id for s in sectors for agent_id in s.agent_ids),
            sorted(positions),
        )
        for sector in sectors:
            self.assertEqual(
                {types[agent_id] for agent_id in sector.agent_ids},
                {"human", "zombie"},
                sector.describe(),
            )
        # The lone zombie is now in an arena that contains at least one human.
        host = next(s for s in sectors if 99 in s.agent_ids)
        self.assertIn("human", {types[agent_id] for agent_id in host.agent_ids})


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_result(self):
        agents = make_agents(60, 60)
        first, _ = make_tasks(agents, 4)
        second, _ = make_tasks(make_agents(60, 60), 4)

        a = [simulate_sector_task(task) for task in copy.deepcopy(first)]
        b = [simulate_sector_task(task) for task in copy.deepcopy(second)]

        self.assertEqual([o.report for o in a], [o.report for o in b])
        self.assertEqual(state_of(first), state_of(second))

    def test_different_seeds_diverge(self):
        # Agents start on top of each other so the seeded combat rolls - not
        # just the layout - decide the outcome.
        base, _ = make_tasks(make_agents(30, 30, seed=1), 2, seed=1)
        other, _ = make_tasks(make_agents(30, 30, seed=5), 2, seed=2)
        for tasks in (base, other):
            for task in tasks:
                for agent in [*(task.humans or []), *(task.zombies or [])]:
                    agent.x_pos = float(agent.agent_id % 7)
                    agent.y_pos = float(agent.agent_id % 5)

        a = [simulate_sector_task(task) for task in copy.deepcopy(base)]
        b = [simulate_sector_task(task) for task in copy.deepcopy(other)]

        self.assertNotEqual([o.report for o in a], [o.report for o in b])


class BackendEquivalenceTests(unittest.TestCase):
    """A sector must behave identically no matter where it is executed."""

    @classmethod
    def setUpClass(cls):
        cls.pristine, _ = make_tasks(make_agents(70, 70), 5)

    def _serial(self):
        tasks = copy.deepcopy(self.pristine)
        outcomes = [simulate_sector_task(task) for task in tasks]
        return tasks, outcomes

    def _pool(self, backend: str, workers: int = 3):
        pool = SectorPool(backend, workers)
        if not pool.active:
            self.skipTest(f"{backend} pool is not available on this machine")
        tasks = copy.deepcopy(self.pristine)
        try:
            outcomes = pool.map(tasks)
        finally:
            pool.close()
        return tasks, outcomes

    def test_thread_pool_matches_serial(self):
        serial_tasks, serial_outcomes = self._serial()
        thread_tasks, thread_outcomes = self._pool("thread")

        self.assertEqual(
            [o.report for o in serial_outcomes], [o.report for o in thread_outcomes]
        )
        self.assertEqual(state_of(serial_tasks), state_of(thread_tasks))

    def test_process_pool_matches_serial(self):
        serial_tasks, serial_outcomes = self._serial()
        process_tasks, process_outcomes = self._pool("process")

        # Worker processes mutate their own copies, so the parent's task
        # payloads are untouched; the reports carry the resulting state
        # (counts, ticks, attacks, stop reason).
        self.assertEqual(
            [o.report for o in serial_outcomes], [o.report for o in process_outcomes]
        )
        self.assertEqual(
            [o.moved for o in serial_outcomes], [o.moved for o in process_outcomes]
        )

    def test_warm_pool_produces_the_same_result(self):
        pool = SectorPool("process", 3)
        if not pool.active:
            self.skipTest("process pool is not available on this machine")
        try:
            pool.warm()  # start workers early, then map as usual
            outcomes = pool.map(copy.deepcopy(self.pristine))
        finally:
            pool.close()

        _, serial_outcomes = self._serial()
        self.assertEqual(
            [o.report for o in serial_outcomes], [o.report for o in outcomes]
        )

    def test_frames_are_identical(self):
        pristine, _ = make_tasks(make_agents(40, 40), 3)
        for task in pristine:
            task.capture_frames = True

        serial = [simulate_sector_task(task) for task in copy.deepcopy(pristine)]
        pool = SectorPool("thread", 2)
        if not pool.active:
            self.skipTest("thread pool is not available on this machine")
        try:
            threaded = pool.map(copy.deepcopy(pristine))
        finally:
            pool.close()

        self.assertEqual(
            [o.frames for o in serial],
            [o.frames for o in threaded],
            "frame data must not depend on the backend",
        )


class StagnationTests(unittest.TestCase):
    """The failsafe that stops runs ending with both sides alive."""

    def test_collapse_once_then_stop(self):
        simulation = Simulation(object())
        self.assertEqual(simulation._stagnation_action(0, 0.0), "collapse")
        simulation._collapse = True
        self.assertEqual(simulation._stagnation_action(0, 0.0), "stop")

    def test_progress_is_not_stagnation(self):
        simulation = Simulation(object())
        self.assertEqual(simulation._stagnation_action(5, 0.0), "")
        self.assertEqual(simulation._stagnation_action(0, 0.001), "")


class PoolSizingTests(unittest.TestCase):
    """Workers execute the sectors the population produced, never more."""

    def test_one_sector_stays_serial(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(12, "process")
        simulation._decide_parallel(1)
        self.assertFalse(simulation._parallel)
        self.assertEqual(simulation.workers, 1)

    def test_workers_are_capped_by_sectors(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(12, "process")
        simulation._decide_parallel(3)
        try:
            if not simulation._parallel:
                self.skipTest("process pool is unavailable on this machine")
            self.assertEqual(simulation.workers, 3)
        finally:
            simulation.close()

    def test_collapse_moves_the_budget_into_the_one_sector(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(8, "thread")
        try:
            simulation._decide_parallel(4)
            simulation._collapse_to_one_arena()
            self.assertFalse(simulation._parallel)
            self.assertEqual(simulation.workers, 1)
            self.assertEqual(simulation.event_shards, 8)
        finally:
            simulation.close()

    def test_collapse_on_a_process_backend_stays_serial(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(8, "process")
        simulation._backend_used = "process"
        simulation._collapse_to_one_arena()
        self.assertEqual(simulation.workers, 1)
        self.assertEqual(simulation.event_shards, 1)
        self.assertEqual(simulation._active_backend(), "serial")


class SeedTests(unittest.TestCase):
    def test_derive_seed_is_stable_and_distinct(self):
        self.assertEqual(derive_seed(7, 3), derive_seed(7, 3))
        self.assertEqual(derive_seed(7, 3, 1), derive_seed(7, 3, 1))
        self.assertNotEqual(derive_seed(7, 3), derive_seed(7, 4))
        self.assertNotEqual(derive_seed(7, 3), derive_seed(8, 3))
        self.assertTrue(0 <= derive_seed(7, 3) < 2**64)


if __name__ == "__main__":
    unittest.main()
