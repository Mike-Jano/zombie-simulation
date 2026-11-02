"""Tick-level event sharding: policy, determinism and the round cache."""

from __future__ import annotations

import copy
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from runtime.capabilities import resolve_event_shards, resolve_plan  # noqa: E402
from src.human import Human  # noqa: E402
from src.sector_engine import derive_seed  # noqa: E402
from src.sector_task import SectorTask, simulate_sector_task  # noqa: E402
from src.simulation import Simulation  # noqa: E402
from src.spawner import build_agents  # noqa: E402
from utils.sectoring import build_sectors  # noqa: E402

SEED = 4242


def make_agents(humans: int, zombies: int, seed: int = SEED) -> list:
    return build_agents(humans, zombies, random.Random(seed))


def make_tasks(agents, sector_count: int, seed: int = SEED, frames: bool = False):
    positions = {
        agent.agent_id: (float(agent.x_pos), float(agent.y_pos)) for agent in agents
    }
    types = {agent.agent_id: agent.agent_type for agent in agents}
    sectors = build_sectors(
        positions, rng=random.Random(seed), target_k=sector_count, types=types
    )
    by_id = {agent.agent_id: agent for agent in agents}
    tasks = []
    for sector in sectors:
        members = [by_id[agent_id] for agent_id in sector.agent_ids]
        tasks.append(
            SectorTask(
                sector=sector,
                seed=derive_seed(seed, sector.index),
                round_index=1,
                sectors_in_round=len(sectors),
                capture_frames=frames,
                humans=[a for a in members if a.agent_type == "human"],
                zombies=[a for a in members if a.agent_type == "zombie"],
            )
        )
    return tasks


def run_shards(humans, zombies, sectors, seed, shards, frames=False):
    tasks = make_tasks(make_agents(humans, zombies, seed), sectors, seed, frames)
    for task in tasks:
        task.event_shards = shards
    outcomes = [simulate_sector_task(task) for task in tasks]
    return {
        "reports": [outcome.report.__dict__ for outcome in outcomes],
        "moved": [outcome.moved for outcome in outcomes],
        "frames": [
            [[list(snapshot), meta] for snapshot, meta in outcome.frames]
            for outcome in outcomes
        ],
        "final": [outcome.final_states for outcome in outcomes],
        "state": {
            agent.agent_id: [
                agent.x_pos,
                agent.y_pos,
                int(agent.health),
                bool(agent.is_alive),
            ]
            for task in tasks
            for agent in [*(task.humans or []), *(task.zombies or [])]
        },
    }


class ShardDeterminismTests(unittest.TestCase):
    def test_shard_count_does_not_change_a_sector(self):
        cases = [
            (40, 40, 3, 4242, True),
            (70, 70, 5, 4242, False),
            (150, 150, 4, 77, False),
            (200, 200, 6, 9, False),
        ]
        for humans, zombies, sectors, seed, frames in cases:
            baseline = run_shards(humans, zombies, sectors, seed, 1, frames)
            for shards in (2, 3, 4, 8, 16):
                with self.subTest(
                    humans=humans,
                    zombies=zombies,
                    sectors=sectors,
                    seed=seed,
                    shards=shards,
                ):
                    self.assertEqual(
                        run_shards(humans, zombies, sectors, seed, shards, frames),
                        baseline,
                    )

    def test_one_agent_sector_handles_extra_shards(self):
        # Degenerate shard counts must not crash or deadlock.
        baseline = run_shards(1, 1, 1, 5, 1, False)
        self.assertEqual(run_shards(1, 1, 1, 5, 32, False), baseline)


class EventShardPolicyTests(unittest.TestCase):
    def setUp(self):
        self._saved = (cfg.EVENT_PARALLEL, cfg.EVENT_SHARDS)

    def tearDown(self):
        cfg.EVENT_PARALLEL, cfg.EVENT_SHARDS = self._saved

    def test_process_and_serial_stay_single_shard(self):
        cfg.EVENT_PARALLEL = "auto"
        cfg.EVENT_SHARDS = 0
        self.assertEqual(resolve_event_shards(1, 8, "process"), 1)
        self.assertEqual(resolve_event_shards(4, 8, "serial"), 1)

    def test_thread_backend_fills_the_worker_budget(self):
        cfg.EVENT_PARALLEL = "auto"
        cfg.EVENT_SHARDS = 0
        self.assertEqual(resolve_event_shards(1, 8, "thread"), 8)
        self.assertEqual(resolve_event_shards(3, 12, "thread"), 4)
        self.assertEqual(resolve_event_shards(12, 12, "thread"), 1)

    def test_explicit_shards_win(self):
        cfg.EVENT_PARALLEL = "auto"
        cfg.EVENT_SHARDS = 0
        self.assertEqual(resolve_event_shards(1, 8, "thread", requested=5), 5)
        self.assertEqual(resolve_event_shards(4, 8, "thread", requested=3), 3)

    def test_explicit_shards_are_capped_by_the_worker_budget(self):
        cfg.EVENT_PARALLEL = "auto"
        cfg.EVENT_SHARDS = 0
        self.assertEqual(resolve_event_shards(1, 8, "thread", requested=1000), 8)
        self.assertEqual(resolve_event_shards(4, 8, "thread", requested=1000), 8)

    def test_off_means_one(self):
        cfg.EVENT_PARALLEL = "off"
        cfg.EVENT_SHARDS = 0
        self.assertEqual(resolve_event_shards(1, 8, "thread"), 1)
        cfg.EVENT_PARALLEL = "auto"
        cfg.EVENT_SHARDS = 6
        self.assertEqual(resolve_event_shards(4, 8, "process"), 1)


class SimulationParallelDecisionTests(unittest.TestCase):
    def test_single_sector_uses_all_shards_without_outer_pool(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(8, "thread")
        try:
            simulation._decide_parallel(1)
            self.assertFalse(simulation._parallel)
            self.assertEqual(simulation.workers, 1)
            self.assertEqual(simulation.event_shards, 8)
        finally:
            simulation.close()

    def test_many_sectors_split_the_budget(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(8, "thread")
        try:
            simulation._decide_parallel(4)
            self.assertTrue(simulation._parallel)
            self.assertEqual(simulation.workers, 4)
            self.assertEqual(simulation.event_shards, 2)
        finally:
            simulation.close()

    def test_process_backend_keeps_events_single_shard(self):
        simulation = Simulation(object())
        simulation.plan = resolve_plan(8, "process")
        try:
            simulation._decide_parallel(4)
            self.assertEqual(simulation.event_shards, 1)
        finally:
            simulation.close()


class ChunkingAndFrameTests(unittest.TestCase):
    """The load-balancing cap and frame interval must not alter the sim."""

    def setUp(self):
        self._saved = (cfg.EVENT_CHUNK_SIZE, cfg.FRAME_INTERVAL)

    def tearDown(self):
        cfg.EVENT_CHUNK_SIZE, cfg.FRAME_INTERVAL = self._saved

    def test_chunk_size_does_not_change_the_sector(self):
        cfg.FRAME_INTERVAL = 1
        cfg.EVENT_CHUNK_SIZE = 0
        baseline = run_shards(120, 120, 3, 4242, 4, False)
        for chunk_size in (1, 2, 7, 50, 1000):
            with self.subTest(chunk_size=chunk_size):
                cfg.EVENT_CHUNK_SIZE = chunk_size
                self.assertEqual(
                    run_shards(120, 120, 3, 4242, 4, False), baseline
                )

    def test_frame_interval_keeps_state_and_shrinks_frames(self):
        cfg.EVENT_CHUNK_SIZE = 0
        cfg.FRAME_INTERVAL = 1
        dense = run_shards(120, 120, 3, 4242, 2, True)
        cfg.FRAME_INTERVAL = 5
        sparse = run_shards(120, 120, 3, 4242, 2, True)

        for key in ("reports", "moved", "final", "state"):
            self.assertEqual(sparse[key], dense[key], key)

        dense_frames = sum(len(frames) for frames in dense["frames"])
        sparse_frames = sum(len(frames) for frames in sparse["frames"])
        self.assertGreater(dense_frames, 0)
        self.assertLess(sparse_frames, dense_frames)

        # Sampled frames must be exact copies of the dense frames at the same
        # iteration, and the closing frame must always survive sampling.
        for dense_sector, sparse_sector in zip(dense["frames"], sparse["frames"]):
            self.assertGreater(len(sparse_sector), 0)
            dense_by_tick = {
                frame[1]["iteration"]: frame
                for frame in dense_sector
                if not frame[1]["ended"]
            }
            ended = [frame for frame in dense_sector if frame[1]["ended"]]
            self.assertEqual(sparse_sector[-1], ended[-1])
            # The first frame is always kept, even when the interval would
            # otherwise skip it.
            self.assertEqual(sparse_sector[0][1]["iteration"], 1)
            for frame in sparse_sector[:-1]:
                self.assertEqual(
                    frame, dense_by_tick[frame[1]["iteration"]]
                )
                self.assertTrue(
                    frame[1]["iteration"] == 1
                    or frame[1]["iteration"] % cfg.FRAME_INTERVAL == 0
                )


# --------------------------------------------------------------------------- #
# Round cache: clustering from outcomes must equal re-reading PostgreSQL.
# --------------------------------------------------------------------------- #
class FakeDB:
    """Minimal DBManager stand-in for an end-to-end round-cache test."""

    def __init__(self, agents):
        self.config: dict = {}
        self.rows = {agent.agent_id: agent for agent in agents}
        self.fetch_calls = 0

    def fetch_alive_agents(self):
        self.fetch_calls += 1
        return [
            (
                agent.agent_id,
                agent.agent_type,
                int(round(agent.x_pos)),
                int(round(agent.y_pos)),
            )
            for agent in sorted(self.rows.values(), key=lambda item: item.agent_id)
            if agent.is_alive
        ]

    def load_agents(self, agent_ids):
        humans, zombies = [], []
        for agent_id in agent_ids:
            agent = self.rows.get(agent_id)
            if agent is None or not agent.is_alive:
                continue
            if isinstance(agent, Human):
                humans.append(agent)
            else:
                zombies.append(agent)
        humans.sort(key=lambda item: item.agent_id)
        zombies.sort(key=lambda item: item.agent_id)
        return humans, zombies

    def save_agents(self, agents):
        for agent in agents:
            self.rows[agent.agent_id] = agent
        return len(agents)


class NoCacheSimulation(Simulation):
    @staticmethod
    def _next_rows(outcomes):
        return None


class RoundCacheTests(unittest.TestCase):
    @staticmethod
    def _agents():
        return make_agents(150, 150, seed=77)

    @staticmethod
    def _summary(report):
        return {
            "rounds": report.rounds,
            "stop_reason": report.stop_reason,
            "humans_end": report.humans_end,
            "zombies_end": report.zombies_end,
            "attacks": report.attacks,
            "iterations": report.iterations,
        }

    @staticmethod
    def _state(db):
        return {
            agent_id: (
                round(agent.x_pos, 9),
                round(agent.y_pos, 9),
                int(agent.health),
                bool(agent.is_alive),
            )
            for agent_id, agent in db.rows.items()
        }

    def test_cached_rows_match_refetching(self):
        cached_db = FakeDB(copy.deepcopy(self._agents()))
        control_db = FakeDB(copy.deepcopy(self._agents()))

        cached = Simulation(
            cached_db,
            rng=random.Random(77),
            sectors=4,
            workers=1,
            backend="serial",
            event_shards=1,
            seed=77,
        )
        control = NoCacheSimulation(
            control_db,
            rng=random.Random(77),
            sectors=4,
            workers=1,
            backend="serial",
            event_shards=1,
            seed=77,
        )

        cached_report = cached.run()
        control_report = control.run()

        self.assertEqual(self._summary(cached_report), self._summary(control_report))
        self.assertEqual(self._state(cached_db), self._state(control_db))
        # One SELECT for the first round; later rounds use the outcome cache.
        self.assertEqual(cached_db.fetch_calls, 1)
        self.assertGreater(control_db.fetch_calls, 1)

    def test_next_rows_filters_dead_and_rounds_like_the_database(self):
        from src.sector_engine import SectorOutcome, SectorReport

        outcome = SectorOutcome(
            report=SectorReport(
                index=1,
                bounds=(0.0, 0.0, 0.0, 0.0),
                humans_start=2,
                zombies_start=0,
                humans_end=1,
                zombies_end=0,
                iterations=1,
                attacks=0,
                stop_reason="test",
            ),
            final_states=[
                (2, "human", 10.6, -3.4, True),
                (1, "human", 99999.0, -99999.0, True),
                (3, "human", 4.2, 4.2, False),
            ],
        )
        rows = Simulation._next_rows([outcome])
        self.assertEqual(
            rows,
            [
                (1, "human", int(cfg.MAP_MAX), int(cfg.MAP_MIN)),
                (2, "human", 11, -3),
            ],
        )


class ReportCountTests(unittest.TestCase):
    def test_one_sided_run_reports_its_population(self):
        db = FakeDB(make_agents(5, 0, seed=1))
        report = Simulation(
            db, rng=random.Random(1), workers=1, backend="serial", seed=1
        ).run()
        self.assertEqual((report.humans_start, report.humans_end), (5, 5))
        self.assertEqual((report.zombies_start, report.zombies_end), (0, 0))
        self.assertIn("Humans : 5 -> 5 alive", report.render())


if __name__ == "__main__":
    unittest.main()
