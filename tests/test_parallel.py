"""Tests for parallel sectoring: policy, determinism and backend equivalence."""

from __future__ import annotations

import copy
import io
import random
import sys
import unittest
from concurrent.futures.process import BrokenProcessPool
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from runtime.capabilities import resolve_plan, resolve_workers  # noqa: E402
from runtime.sector_pool import SectorPool  # noqa: E402
from src.sector_engine import SectorEngine, derive_seed  # noqa: E402
from src.sector_task import build_sector_tasks, simulate_sector_task  # noqa: E402
from src.simulation import Simulation  # noqa: E402
from src.spawner import build_agents  # noqa: E402
from utils.sectoring import (  # noqa: E402
    build_sectors,
    max_sectors_for,
    resolve_sector_count,
)

SEED = 4242


def make_agents(humans: int, zombies: int, seed: int = SEED) -> list:
    return build_agents(humans, zombies, random.Random(seed))


def make_tasks(agents, sector_count: int, seed: int = SEED):
    return build_sector_tasks(agents, seed, sector_count)


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
        # 500 agents per sector, with no fixed upper limit.
        self.assertEqual(resolve_sector_count(200), 1)
        self.assertEqual(resolve_sector_count(500), 1)
        self.assertEqual(resolve_sector_count(501), 2)
        self.assertEqual(resolve_sector_count(1500), 3)
        self.assertEqual(resolve_sector_count(3000), 6)
        self.assertEqual(resolve_sector_count(6000), 12)
        self.assertEqual(resolve_sector_count(60_000), 120)

    def test_max_sectors_scales_with_the_starting_population(self):
        self.assertEqual(max_sectors_for(0), 1)
        self.assertEqual(max_sectors_for(200), 1)
        self.assertEqual(max_sectors_for(501), 2)
        self.assertEqual(max_sectors_for(6000), 12)
        self.assertEqual(max_sectors_for(100_000), 200)

    def test_sectors_fall_as_the_living_population_falls(self):
        # The ceiling is fixed for the run; the actual count follows the living.
        ceiling = max_sectors_for(3000)
        self.assertEqual(ceiling, 6)
        self.assertEqual(resolve_sector_count(3000, max_sectors=ceiling), 6)
        self.assertEqual(resolve_sector_count(1500, max_sectors=ceiling), 3)
        self.assertEqual(resolve_sector_count(100, max_sectors=ceiling), 1)

    def test_worker_count_does_not_change_the_plan(self):
        # There is no worker argument any more: cores cannot reshape the run.
        self.assertEqual(resolve_sector_count(200), 1)
        self.assertEqual(resolve_sector_count(10), 1)
        self.assertEqual(resolve_sector_count(0), 0)

    def test_explicit_request_wins(self):
        self.assertEqual(resolve_sector_count(1200, requested=3), 3)
        # ...but never beyond the run's population-derived ceiling.
        self.assertEqual(resolve_sector_count(1200, requested=99), 3)

    def test_build_sectors_honours_target(self):
        agents = make_agents(200, 200)
        positions = {a.agent_id: (a.x_pos, a.y_pos) for a in agents}
        types = {a.agent_id: a.agent_type for a in agents}
        sectors = build_sectors(
            positions, target_k=6, types=types
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
            positions, target_k=10, types=types
        )
        for sector in sectors:
            kinds = {types[agent_id] for agent_id in sector.agent_ids}
            self.assertEqual(kinds, {"human", "zombie"}, repr(sector))

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
            positions, target_k=4, types=types
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
            positions, target_k=3, types=types
        )
        self.assertEqual(
            sorted(agent_id for s in sectors for agent_id in s.agent_ids),
            sorted(positions),
        )
        for sector in sectors:
            self.assertEqual(
                {types[agent_id] for agent_id in sector.agent_ids},
                {"human", "zombie"},
                repr(sector),
            )
        # The lone zombie is now in an arena that contains at least one human.
        host = next(s for s in sectors if 99 in s.agent_ids)
        self.assertIn("human", {types[agent_id] for agent_id in host.agent_ids})


class DeterminismTests(unittest.TestCase):
    def test_same_seed_same_result(self):
        agents = make_agents(60, 60)
        first = make_tasks(agents, 4)
        second = make_tasks(make_agents(60, 60), 4)

        a = [simulate_sector_task(task) for task in copy.deepcopy(first)]
        b = [simulate_sector_task(task) for task in copy.deepcopy(second)]

        self.assertEqual([o.report for o in a], [o.report for o in b])
        self.assertEqual(state_of(first), state_of(second))

    def test_different_seeds_diverge(self):
        # Agents start on top of each other so the seeded combat rolls - not
        # just the layout - decide the outcome.
        base = make_tasks(make_agents(30, 30, seed=1), 2, seed=1)
        other = make_tasks(make_agents(30, 30, seed=5), 2, seed=2)
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
        cls.pristine = make_tasks(make_agents(70, 70), 5)

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
        pristine = make_tasks(make_agents(40, 40), 3)
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


class PoolWidthTests(unittest.TestCase):
    """The pool must shrink when the living population halves."""

    def test_first_round_decides_the_pool(self):
        simulation = Simulation(object())
        with patch.object(simulation, "_decide_parallel") as decide:
            simulation._sync_pool_width(4)
            decide.assert_called_once_with(4)

    def test_pool_shrinks_when_the_sector_count_halves(self):
        simulation = Simulation(object())
        simulation._parallel = True
        simulation.workers = 6
        with patch.object(simulation, "_decide_parallel") as decide:
            simulation._sync_pool_width(3)
            decide.assert_called_once_with(3)
            decide.reset_mock()
            simulation._sync_pool_width(4)
            decide.assert_not_called()


class SeedTests(unittest.TestCase):
    def test_derive_seed_is_stable_and_distinct(self):
        self.assertEqual(derive_seed(7, 3), derive_seed(7, 3))
        self.assertEqual(derive_seed(7, 3, 1), derive_seed(7, 3, 1))
        self.assertNotEqual(derive_seed(7, 3), derive_seed(7, 4))
        self.assertNotEqual(derive_seed(7, 3), derive_seed(8, 3))
        self.assertTrue(0 <= derive_seed(7, 3) < 2**64)


class WorkerCapTests(unittest.TestCase):
    def test_explicit_worker_count_is_capped(self):
        self.assertLessEqual(resolve_workers(100_000), cfg.MAX_WORKERS)


class TickBudgetTests(unittest.TestCase):
    def test_zero_refresh_ticks_still_gives_one_tick_per_round(self):
        self.assertEqual(cfg._tick_budget(0), cfg.MAX_ROUNDS)
        self.assertEqual(
            cfg._tick_budget(cfg.SECTOR_REFRESH_TICKS), cfg.MAX_TOTAL_TICKS
        )

    def test_tick_budget_uses_the_hard_per_visit_cap(self):
        self.assertEqual(
            cfg._tick_budget(cfg.MAX_ITERATIONS_PER_SECTOR + 1),
            cfg.MAX_ROUNDS * cfg.MAX_ITERATIONS_PER_SECTOR,
        )

    def test_zero_refresh_ticks_still_runs_one_engine_tick(self):
        with patch.object(cfg, "SECTOR_REFRESH_TICKS", 0):
            self.assertEqual(SectorEngine().max_ticks, 1)


class BenchToolTests(unittest.TestCase):
    def test_population_above_the_limit_is_rejected(self):
        from tools import bench_parallel

        with redirect_stdout(io.StringIO()):
            code = bench_parallel.main(["--population", str(cfg.MAX_AGENTS + 1)])
        self.assertEqual(code, 1)

    def test_zero_population_is_rejected(self):
        from tools import bench_parallel

        with redirect_stdout(io.StringIO()):
            code = bench_parallel.main(["--population", "0"])
        self.assertEqual(code, 1)

    def test_split_outside_a_fraction_is_rejected(self):
        from tools import bench_parallel

        with redirect_stdout(io.StringIO()):
            code = bench_parallel.main(["--population", "10", "--split", "2"])
        self.assertEqual(code, 1)

    def test_bad_worker_list_is_rejected(self):
        from tools import bench_parallel

        with redirect_stdout(io.StringIO()):
            code = bench_parallel.main(["--workers", "1,eight"])
        self.assertEqual(code, 1)

    def test_rows_use_the_same_budget_and_shards_as_a_real_run(self):
        from tools import bench_parallel

        # 8 workers, 3 sectors: 3 outer + 2 shards each (thread backend).
        self.assertEqual(bench_parallel._row_plan(8, 3, "thread", 0), (8, 3, 2))
        # 100 workers is capped at MAX_WORKERS, and process runs never shard.
        self.assertEqual(
            bench_parallel._row_plan(100, 200, "process", 0), (cfg.MAX_WORKERS, 12, 1)
        )
        self.assertEqual(bench_parallel._row_plan(6, 3, "process", 4), (6, 3, 1))


class ReportWidthTests(unittest.TestCase):
    def test_report_keeps_the_widest_width_when_the_pool_shrinks(self):
        simulation = Simulation(SimpleNamespace(config={}))
        simulation.report.workers = 6
        simulation.report.event_shards = 3
        sector = SimpleNamespace(index=1, agent_ids=[], bounds=(0.0, 0.0, 1.0, 1.0))
        outcome = SimpleNamespace(report=SimpleNamespace(index=1))

        simulation.workers = 8
        simulation.event_shards = 4
        with patch.object(simulation, "_run_sector_serial", return_value=outcome):
            simulation._run_round([sector], 1, 123)
        self.assertEqual(simulation.report.workers, 8)
        self.assertEqual(simulation.report.event_shards, 4)

        simulation.workers = 2
        simulation.event_shards = 1
        with patch.object(simulation, "_run_sector_serial", return_value=outcome):
            simulation._run_round([sector], 2, 456)
        self.assertEqual(simulation.report.workers, 8)
        self.assertEqual(simulation.report.event_shards, 4)


class _FailingPool:
    active = True
    backend = "process"

    def map(self, tasks):
        raise BrokenProcessPool("boom")

    def close(self):
        pass


class PoolFailureTests(unittest.TestCase):
    def test_pool_failure_stops_instead_of_continuing(self):
        simulation = Simulation(SimpleNamespace(config={}))
        simulation._pool = _FailingPool()
        simulation._parallel = True
        simulation.workers = 6
        simulation.event_shards = 2
        simulation._backend_used = "process"
        sector = SimpleNamespace(index=1, agent_ids=[], bounds=(0.0, 0.0, 1.0, 1.0))
        with self.assertRaises(RuntimeError):
            simulation._run_round([sector], 1, 123)

        # The report keeps the width that was actually used, not the serial
        # fallback installed by _disable_pool.
        self.assertEqual(simulation.report.workers, 6)
        self.assertEqual(simulation.report.backend, "process")
        self.assertEqual(simulation.report.event_shards, 2)


if __name__ == "__main__":
    unittest.main()
