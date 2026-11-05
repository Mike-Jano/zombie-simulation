"""The map is sized from the starting population; the schema has no fixed box."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from db import schema  # noqa: E402
from db.db_manager import _clamp_int  # noqa: E402
from src.spawner import generate_spawns  # noqa: E402
from utils.sectoring import map_bounds_for  # noqa: E402
from vis.html_vis import _grid_step  # noqa: E402


def _expected_half(count: int) -> int:
    raw = max(
        int(cfg.MAP_MIN_HALF_EXTENT),
        int(math.ceil(math.sqrt(count * cfg.MAP_AREA_PER_AGENT) / 2.0)),
    )
    step = int(cfg.MAP_SIZE_STEP)
    return max(step, ((raw + step // 2) // step) * step)


class DynamicBoundsTests(unittest.TestCase):
    def test_minimum_map_uses_the_snapped_minimum(self):
        half = _expected_half(0)
        expected = (-half, -half, half, half)
        self.assertEqual(map_bounds_for(0), expected)
        self.assertEqual(map_bounds_for(1), expected)
        self.assertEqual(map_bounds_for(10), expected)

    def test_map_grows_in_whole_grid_steps(self):
        previous = 0
        for count in (10, 80, 500, 1500, 3000, 6000, 60_000):
            half = map_bounds_for(count)[2]
            self.assertGreaterEqual(half, previous)
            self.assertGreaterEqual(half, cfg.MAP_MIN_HALF_EXTENT)
            previous = half

    def test_bounds_are_integers_on_the_grid(self):
        for count in (0, 1, 10, 80, 500, 1500, 3000, 6000, 60_000):
            bounds = map_bounds_for(count)
            with self.subTest(count=count):
                self.assertTrue(all(isinstance(value, int) for value in bounds))
                self.assertEqual(bounds[0], -bounds[2])
                self.assertEqual(bounds[1], -bounds[3])
                self.assertEqual(bounds[2] % cfg.MAP_SIZE_STEP, 0)
                self.assertEqual(bounds[2], _expected_half(count))

    def test_a_raw_2733_map_snaps_to_2700_then_2800(self):
        self.assertEqual(map_bounds_for(29_860)[2], 2700)
        self.assertEqual(map_bounds_for(30_230)[2], 2800)

    def test_four_thousand_agents_get_a_two_thousand_wide_map(self):
        self.assertEqual(map_bounds_for(4000), (-1000, -1000, 1000, 1000))


class SchemaBoundsTests(unittest.TestCase):
    def test_schema_does_not_bake_in_a_fixed_map(self):
        self.assertNotIn("agents_x_pos_bounds", schema.AGENTS_DDL)
        self.assertNotIn("agents_y_pos_bounds", schema.AGENTS_DDL)
        self.assertNotIn("BETWEEN", schema.AGENTS_DDL)

    def test_map_bound_migration_helper_exists(self):
        self.assertTrue(callable(getattr(schema, "_drop_map_bound_checks", None)))


class ClampTests(unittest.TestCase):
    def test_db_rounding_clamps_to_the_run_bounds(self):
        bounds = (-1000.0, -1000.0, 1000.0, 1000.0)
        self.assertEqual(_clamp_int(1000.4, bounds), 1000)
        self.assertEqual(_clamp_int(1000.6, bounds), 1000)
        self.assertEqual(_clamp_int(-1000.6, bounds), -1000)
        self.assertEqual(_clamp_int(500.4, bounds), 500)

    def test_without_bounds_only_rounds(self):
        self.assertEqual(_clamp_int(5000.4), 5000)


class GridStepTests(unittest.TestCase):
    def test_grid_is_roughly_twenty_cells(self):
        for span in (100, 200, 1000, 2000, 5000, 10_000):
            step = _grid_step(span)
            cells = span // step
            with self.subTest(span=span):
                self.assertGreaterEqual(cells, 10)
                self.assertLessEqual(cells, 25)


class SpawnBoundsTests(unittest.TestCase):
    def test_spawns_cover_the_dynamic_map(self):
        specs = generate_spawns(2000, 2000, random.Random(1))
        bounds = map_bounds_for(4000)
        xs = [spec["x_pos"] for spec in specs]
        ys = [spec["y_pos"] for spec in specs]

        self.assertTrue(all(bounds[0] <= x <= bounds[2] for x in xs))
        self.assertTrue(all(bounds[1] <= y <= bounds[3] for y in ys))
        self.assertLess(min(xs), -900)
        self.assertGreater(max(xs), 900)
        self.assertLess(min(ys), -900)
        self.assertGreater(max(ys), 900)

    def test_spawner_uses_explicit_bounds(self):
        specs = generate_spawns(50, 0, random.Random(0), (-5, -5, 5, 5))
        for spec in specs:
            self.assertTrue(-5 <= spec["x_pos"] <= 5, spec)
            self.assertTrue(-5 <= spec["y_pos"] <= 5, spec)


if __name__ == "__main__":
    unittest.main()
