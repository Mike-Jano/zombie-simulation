"""Spatial index must agree with the brute-force scan, including ties."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.spatial import SpatialIndex, brute_force  # noqa: E402


def agent(agent_id: int, x: float, y: float, alive: bool = True):
    return SimpleNamespace(
        agent_id=agent_id,
        x_pos=float(x),
        y_pos=float(y),
        is_alive=alive,
    )


def assert_same(test, index, agents, x, y, exclude_id=None):
    expected_agent, expected_distance = brute_force(
        agents, x, y, exclude_id=exclude_id
    )
    got_agent, got_distance = index.nearest(x, y, exclude_id=exclude_id)
    test.assertIs(got_agent, expected_agent)
    if expected_agent is None:
        test.assertEqual(got_distance, float("inf"))
    else:
        test.assertAlmostEqual(got_distance, expected_distance, places=12)


class SpatialIndexTests(unittest.TestCase):
    def test_random_layouts_match_brute_force(self):
        rng = random.Random(1234)
        for _ in range(20):
            count = rng.randint(1, 120)
            agents = [
                agent(
                    i + 1,
                    rng.uniform(-500, 500),
                    rng.uniform(-500, 500),
                    alive=rng.random() > 0.1,
                )
                for i in range(count)
            ]
            index = SpatialIndex(agents)
            for _ in range(40):
                x = rng.uniform(-600, 600)
                y = rng.uniform(-600, 600)
                assert_same(self, index, agents, x, y)

    def test_grid_layout_ties_pick_lowest_id(self):
        # Four enemies exactly one unit away in each direction.  The old loop
        # picked the first candidate (ascending id), so the index must too.
        agents = [
            agent(4, 0, 1),
            agent(3, 0, -1),
            agent(2, 1, 0),
            agent(1, -1, 0),
            agent(9, 100, 100),
        ]
        index = SpatialIndex(agents)
        winner, distance = index.nearest(0, 0)
        self.assertEqual(winner.agent_id, 1)
        self.assertAlmostEqual(distance, 1.0, places=12)

    def test_exact_duplicate_positions_tie_on_id(self):
        agents = [agent(7, 5, 5), agent(2, 5, 5), agent(5, 5, 5)]
        index = SpatialIndex(agents)
        winner, distance = index.nearest(5, 5)
        self.assertEqual(winner.agent_id, 2)
        self.assertEqual(distance, 0.0)

    def test_dead_agents_are_ignored(self):
        agents = [
            agent(1, 0, 0, alive=False),
            agent(2, 10, 0, alive=True),
            agent(3, 20, 0, alive=True),
        ]
        index = SpatialIndex(agents)
        self.assertEqual(len(index), 2)
        winner, distance = index.nearest(0, 0)
        self.assertEqual(winner.agent_id, 2)
        self.assertAlmostEqual(distance, 10.0)

    def test_exclude_id(self):
        agents = [agent(1, 1, 0), agent(2, 2, 0)]
        index = SpatialIndex(agents)
        winner, _ = index.nearest(1, 0, exclude_id=1)
        self.assertEqual(winner.agent_id, 2)
        self.assertEqual(index.nearest(0, 0, exclude_id=None)[0].agent_id, 1)

    def test_empty_index(self):
        index = SpatialIndex([])
        winner, distance = index.nearest(0, 0)
        self.assertIsNone(winner)
        self.assertEqual(distance, float("inf"))

    def test_collinear_and_duplicate_points(self):
        agents = [agent(i + 1, float(i % 7), float((i * 3) % 5)) for i in range(80)]
        index = SpatialIndex(agents)
        for x in range(-5, 12):
            for y in range(-5, 10):
                assert_same(self, index, agents, x, y)

    def test_bounds_supplied_and_query_outside(self):
        agents = [agent(1, 0, 0), agent(2, 10, 10), agent(3, -10, 5)]
        bounds = (-100.0, -100.0, 100.0, 100.0)
        index = SpatialIndex(agents, bounds=bounds)
        for x in (-200, -100, 0, 50, 100, 200):
            for y in (-200, -100, 0, 50, 100, 200):
                assert_same(self, index, agents, x, y)

    def test_agents_outside_bounds_remain_findable(self):
        # A caller can pass tight sector bounds; an agent outside them must
        # still be targetable instead of silently disappearing.
        agents = [agent(1, 0, 0), agent(2, 150, 150)]
        bounds = (-100.0, -100.0, 100.0, 100.0)
        index = SpatialIndex(agents, bounds=bounds)
        assert_same(self, index, agents, 90, 90)
        self.assertEqual(index.nearest(90, 90)[0].agent_id, 2)


if __name__ == "__main__":
    unittest.main()
