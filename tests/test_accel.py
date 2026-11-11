"""Every accelerated backend must answer exactly like the scalar grid.

The expensive property here is not speed but sameness: a seeded run has to
produce identical rows, frames and report lines whichever backend ran it, so
these tests compare indices, distances and whole sector visits across
``python``, ``numpy`` and (when the machine has one) ``gpu``.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from runtime import accelerator as accel_mod  # noqa: E402
from src.human import Human  # noqa: E402
from src.sector_engine import _plan_attacks, _plan_attacks_accel  # noqa: E402
from src.sector_task import build_sector_tasks, simulate_sector_task  # noqa: E402
from src.spawner import build_agents  # noqa: E402
from src.zombie import Zombie  # noqa: E402
from utils import spatial_batch  # noqa: E402
from utils.spatial import SpatialIndex  # noqa: E402
from utils.sectoring import resolve_sector_count  # noqa: E402


def _points(count, rng, low=-500.0, high=500.0):
    return [(rng.uniform(low, high), rng.uniform(low, high)) for _ in range(count)]


def _brute_index(targets, point):
    """Nearest position in ``targets``; ties keep the lowest index."""
    best, best_d2 = -1, math.inf
    for index, (tx, ty) in enumerate(targets):
        dx = tx - point[0]
        dy = ty - point[1]
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best, best_d2 = index, d2
    return best


def has_gpu() -> bool:
    adapter, _ = accel_mod._gpu_adapter()
    return adapter is not None


class SpatialBatchTests(unittest.TestCase):
    def test_random_layouts_match_brute_force(self):
        rng = random.Random(1234)
        for _ in range(20):
            queries = _points(rng.randint(1, 60), rng)
            targets = _points(rng.randint(1, 60), rng)
            got = spatial_batch.nearest_indices(queries, targets)
            self.assertEqual(
                got, [_brute_index(targets, query) for query in queries]
            )

    def test_ties_and_duplicates_pick_lowest_index(self):
        targets = [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 3.0)]
        self.assertEqual(
            spatial_batch.nearest_indices([(0.0, 0.0)] * 5, targets), [0] * 5
        )
        stacked = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
        self.assertEqual(spatial_batch.nearest_indices([(5.0, 5.0)], stacked), [0])

    def test_exact_half_tie(self):
        # Both targets are exactly sqrt(200) away; the first one must win.
        got = spatial_batch.nearest_indices([(10.0, 10.0)] * 8, [(0.0, 0.0), (20.0, 20.0)])
        self.assertEqual(got, [0] * 8)

    def test_empty_sides(self):
        self.assertEqual(spatial_batch.nearest_indices([], [(0.0, 0.0)]), [])
        self.assertEqual(spatial_batch.nearest_indices([(1.0, 2.0)], []), [-1])

    def test_tiling_does_not_change_the_answer(self):
        rng = random.Random(99)
        queries = _points(37, rng)
        targets = _points(53, rng)
        whole = spatial_batch.nearest_indices(queries, targets)
        with mock.patch.object(cfg, "NUMPY_TILE_ELEMS", 7):
            tiled = spatial_batch.nearest_indices(queries, targets)
        self.assertEqual(whole, tiled)


class AcceleratorAgreementTests(unittest.TestCase):
    """numpy and the scalar stand-in must agree, bit for bit, on distances."""

    def test_python_backend_matches_grid(self):
        rng = random.Random(7)
        queries = _points(40, rng)
        targets = _points(35, rng)
        index = SpatialIndex(
            [
                type("A", (), {"x_pos": x, "y_pos": y, "is_alive": True, "agent_id": i})()
                for i, (x, y) in enumerate(targets)
            ]
        )
        backend = accel_mod.PythonAccelerator()
        indices, distances = backend.nearest_cross(queries, targets)
        for query, index_, distance in zip(queries, indices, distances):
            winner, expected = index.nearest(*query)
            self.assertEqual(index_, winner.agent_id)
            self.assertEqual(distance, expected)

    def test_numpy_backend_matches_grid(self):
        rng = random.Random(8)
        queries = _points(120, rng)
        targets = _points(90, rng)
        index = SpatialIndex(
            [
                type("A", (), {"x_pos": x, "y_pos": y, "is_alive": True, "agent_id": i})()
                for i, (x, y) in enumerate(targets)
            ]
        )
        indices, distances = accel_mod.NumpyAccelerator().nearest_cross(
            queries, targets
        )
        for query, index_, distance in zip(queries, indices, distances):
            winner, expected = index.nearest(*query)
            self.assertEqual(index_, winner.agent_id)
            self.assertEqual(distance, expected)

    def test_numpy_empty_sides(self):
        backend = accel_mod.NumpyAccelerator()
        self.assertEqual(backend.nearest_cross([], [(0.0, 0.0)]), ([], []))
        self.assertEqual(
            backend.nearest_cross([(1.0, 2.0)], []), ([-1], [float("inf")])
        )


class CanonicalOrderTests(unittest.TestCase):
    """Task payloads must be id-ordered: the batched path breaks ties by position."""

    def test_built_tasks_are_in_ascending_id_order(self):
        # A mono-type cluster (1..3, humans) is appended to the mixed cluster
        # (10..13), so the sector's agent_ids are not sorted.
        agents = [
            Human(10, "human", True, -400.0, -400.0, 100, 10, 1),
            Zombie(11, "zombie", True, -400.0, -399.0, 100, 5, 5),
            Human(12, "human", True, -400.0, -398.0, 100, 10, 1),
            Zombie(13, "zombie", True, -400.0, -397.0, 100, 5, 5),
            Human(1, "human", True, 400.0, 400.0, 100, 10, 1),
            Human(2, "human", True, 400.0, 401.0, 100, 10, 1),
            Human(3, "human", True, 400.0, 402.0, 100, 10, 1),
        ]
        tasks = build_sector_tasks(agents, 1, 2)
        self.assertTrue(any(task.humans and task.zombies for task in tasks))
        self.assertTrue(
            any(
                task.sector.agent_ids != sorted(task.sector.agent_ids)
                for task in tasks
            ),
            "fixture no longer exercises mono-type merging",
        )
        for task in tasks:
            for group in (task.humans, task.zombies):
                ids = [agent.agent_id for agent in group or []]
                self.assertEqual(ids, sorted(ids), f"sector {task.sector.index}: {ids}")

    def test_plans_agree_on_an_exact_tie(self):
        humans = [
            Human(1, "human", True, 0.0, 10.0, 100, 10, 1),
            Human(10, "human", True, 0.0, -10.0, 100, 10, 1),
        ]
        zombie = Zombie(11, "zombie", True, 0.0, 0.0, 100, 5, 5)
        scalar = _plan_attacks([zombie], SpatialIndex(humans))
        with mock.patch.object(cfg, "ACCEL_MIN_PAIRS", 1):
            batched = _plan_attacks_accel(
                [zombie], humans, accel_mod.NumpyAccelerator()
            )
        self.assertEqual([plan.target_id for plan in scalar], [1])
        self.assertEqual([plan.target_id for plan in batched], [1])


class BandCheckTests(unittest.TestCase):
    """The f32 safety band is the only thing standing between a GPU and f64."""

    def test_near_tie_is_flagged(self):
        # Real failure found on a Radeon 840M: two zombies 1.0 and
        # 0.99999999999997 away, f32 rounded them the wrong way round because
        # the query sat at (280, -334).
        self.assertTrue(accel_mod._needs_exact(1.0, 0.9999999999999705, 334.0))

    def test_clear_winner_is_not_flagged(self):
        self.assertFalse(accel_mod._needs_exact(106369.0, 108352.0, 500.0))
        self.assertFalse(accel_mod._needs_exact(4.0, 400.0, 10.0))

    def test_identical_distances_are_flagged(self):
        self.assertTrue(accel_mod._needs_exact(2.5, 2.5, 0.0))


class ProbeTests(unittest.TestCase):
    def test_off_is_pure_python(self):
        self.assertEqual(accel_mod.probe("off").mode, "python")

    def test_numpy_request(self):
        self.assertEqual(accel_mod.probe("numpy").mode, "numpy")

    def test_auto_never_starts_the_gpu_stack(self):
        self.assertIn(accel_mod.probe("auto").mode, ("numpy", "python"))

    def test_gpu_request_without_adapter_falls_back_to_numpy(self):
        with mock.patch.object(
            accel_mod, "_gpu_adapter", lambda: (None, "no usable adapter")
        ):
            plan = accel_mod.probe("gpu")
        self.assertEqual(plan.mode, "numpy")
        self.assertIn("no usable adapter", plan.detail)

    def test_gpu_request_without_numpy_keeps_python(self):
        with mock.patch.object(accel_mod, "_numpy_available", lambda: False):
            plan = accel_mod.probe("gpu")
        self.assertEqual(plan.mode, "python")

    def test_describe_names_the_backend(self):
        self.assertIn("gpu", accel_mod.AccelPlan("gpu", "Fake GPU").describe())
        self.assertIn("numpy", accel_mod.AccelPlan("numpy").describe())


@unittest.skipUnless(has_gpu(), "no WebGPU adapter on this machine")
class GpuAgreementTests(unittest.TestCase):
    def setUp(self):
        # Skip the speed calibration: this test is about agreement.
        patcher = mock.patch.object(accel_mod, "_calibrate", lambda gpu, numpy: True)
        patcher.start()
        self.addCleanup(patcher.stop)
        accel_mod.configure("gpu")
        backend = accel_mod.get()
        if getattr(backend, "name", "") != "gpu":
            self.skipTest("GPU device could not be created")
        self.backend = backend
        self.numpy = accel_mod.NumpyAccelerator()

    def test_random_batches_match_numpy(self):
        rng = random.Random(11)
        for query_count, target_count in ((300, 300), (37, 900), (500, 250)):
            queries = _points(query_count, rng)
            targets = _points(target_count, rng)
            self.assertEqual(
                self.backend.nearest_cross(queries, targets),
                self.numpy.nearest_cross(queries, targets),
            )

    def test_near_ties_and_duplicates_match_numpy(self):
        queries = [(280.0, -334.0)] * 4 + [(0.0, 0.0)] * 4
        targets = [
            (280.73715414020074, -333.32427537148266),
            (279.71265211443364, -334.95782628522113),
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        ]
        self.assertEqual(
            self.backend.nearest_cross(queries, targets),
            self.numpy.nearest_cross(queries, targets),
        )


class DeterminismTests(unittest.TestCase):
    """A seeded sector visit is byte-identical across backends."""

    def _visit(self, mode: str, population: int = 700, seed: int = 7):
        agents = build_agents(population // 2, population // 2, random.Random(seed))
        count = resolve_sector_count(len(agents))
        self.assertGreater(count, 1, "keep the per-sector seed path under test")
        tasks = build_sector_tasks(
            agents, seed, count, capture_frames=True, accel=mode
        )
        outcomes = [simulate_sector_task(task) for task in tasks]
        result = [
            (
                outcome.report.iterations,
                outcome.report.attacks,
                outcome.report.stop_reason,
                outcome.final_states,
                outcome.frames,
            )
            for outcome in outcomes
        ]
        self.assertTrue(result)
        self.assertTrue(any(row[0] > 0 for row in result), "the visit did no work")
        return result

    def test_task_carries_the_requested_mode(self):
        agents = build_agents(50, 50, random.Random(3))
        tasks = build_sector_tasks(
            agents, 3, resolve_sector_count(len(agents)), accel="numpy"
        )
        self.assertTrue(tasks)
        self.assertTrue(all(task.accel == "numpy" for task in tasks))

    def test_python_and_numpy_visits_are_identical(self):
        self.assertEqual(self._visit(""), self._visit("numpy"))

    @unittest.skipUnless(has_gpu(), "no WebGPU adapter on this machine")
    def test_gpu_visits_are_identical_to_numpy(self):
        reference = self._visit("numpy")
        with mock.patch.object(accel_mod, "_calibrate", lambda gpu, numpy: True):
            got = self._visit("gpu")
        self.assertEqual(reference, got)


if __name__ == "__main__":
    unittest.main()
