"""Map bounds live in one place and the schema, spawner and player follow them."""

from __future__ import annotations

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
from src.agent import Agent, clamp_map  # noqa: E402
from src.spawner import generate_spawns  # noqa: E402
from vis.html_vis import _grid_step  # noqa: E402

EXPECTED_MIN = -1000
EXPECTED_MAX = 1000


class ConfigBoundsTests(unittest.TestCase):
    def test_map_is_two_thousand_units_across(self):
        self.assertEqual(cfg.MAP_MIN, EXPECTED_MIN)
        self.assertEqual(cfg.MAP_MAX, EXPECTED_MAX)
        self.assertEqual(cfg.MAP_SPAN, EXPECTED_MAX - EXPECTED_MIN)


class SchemaBoundsTests(unittest.TestCase):
    def test_ddl_uses_configured_bounds(self):
        for column, constraint in (
            ("x_pos", "agents_x_pos_bounds"),
            ("y_pos", "agents_y_pos_bounds"),
        ):
            self.assertIn(constraint, schema.AGENTS_DDL)
            self.assertIn(
                f"CHECK ({column} BETWEEN {cfg.MAP_MIN} AND {cfg.MAP_MAX})",
                schema.AGENTS_DDL,
            )

    def test_checkdef_parsing(self):
        cases = {
            # What Postgres actually returns for the two schema versions.
            "CHECK (((x_pos >= '-1000'::integer) AND (x_pos <= 1000)))": (-1000, 1000),
            "CHECK (((y_pos >= '-500'::integer) AND (y_pos <= 500)))": (-500, 500),
            "CHECK ((x_pos >= 0) AND (x_pos <= 1000))": (0, 1000),
            "CHECK ((x_pos >= 1000) AND (x_pos <= 1000))": None,
            "CHECK ((is_alive))": None,
        }
        for definition, expected in cases.items():
            with self.subTest(definition=definition):
                self.assertEqual(schema._bounds_from_checkdef(definition), expected)

    def test_old_bounds_are_recognised_as_stale(self):
        old = "CHECK (((x_pos >= '-500'::integer) AND (x_pos <= 500)))"
        self.assertNotEqual(
            schema._bounds_from_checkdef(old), (cfg.MAP_MIN, cfg.MAP_MAX)
        )


class ClampTests(unittest.TestCase):
    def test_clamp_map_edges(self):
        self.assertEqual(clamp_map(5000), float(cfg.MAP_MAX))
        self.assertEqual(clamp_map(-5000), float(cfg.MAP_MIN))
        self.assertEqual(clamp_map(500), 500.0)  # valid mid-map value

    def test_agent_post_init_clamps(self):
        agent = Agent(1, "human", True, 9999.0, -9999.0, 10, 1)
        self.assertEqual(agent.x_pos, float(cfg.MAP_MAX))
        self.assertEqual(agent.y_pos, float(cfg.MAP_MIN))

    def test_db_rounding_clamps(self):
        self.assertEqual(_clamp_int(1000.4), 1000)
        self.assertEqual(_clamp_int(1000.6), 1000)
        self.assertEqual(_clamp_int(-1000.6), -1000)
        self.assertEqual(_clamp_int(500.4), 500)


class GridStepTests(unittest.TestCase):
    def test_current_map_grid_is_one_hundred_units(self):
        self.assertEqual(_grid_step(cfg.MAP_SPAN), 100)
        self.assertEqual(cfg.MAP_SPAN % _grid_step(cfg.MAP_SPAN), 0)

    def test_grid_is_roughly_twenty_cells(self):
        for span in (200, 1000, 2000, 5000):
            step = _grid_step(span)
            cells = span // step
            with self.subTest(span=span):
                self.assertGreaterEqual(cells, 10)
                self.assertLessEqual(cells, 25)


class SpawnBoundsTests(unittest.TestCase):
    def test_spawns_cover_the_whole_map(self):
        specs = generate_spawns(2000, 2000, random.Random(1))
        xs = [spec["x_pos"] for spec in specs]
        ys = [spec["y_pos"] for spec in specs]

        self.assertTrue(all(cfg.MAP_MIN <= x <= cfg.MAP_MAX for x in xs))
        self.assertTrue(all(cfg.MAP_MIN <= y <= cfg.MAP_MAX for y in ys))
        # The sample has to reach outside the old +-500 box on both axes.
        self.assertLess(min(xs), -900)
        self.assertGreater(max(xs), 900)
        self.assertLess(min(ys), -900)
        self.assertGreater(max(ys), 900)

    def test_spawner_reads_config_at_call_time(self):
        original = (cfg.MAP_MIN, cfg.MAP_MAX)
        try:
            cfg.MAP_MIN, cfg.MAP_MAX = -5, 5
            specs = generate_spawns(50, 0, random.Random(0))
        finally:
            cfg.MAP_MIN, cfg.MAP_MAX = original

        for spec in specs:
            self.assertTrue(-5 <= spec["x_pos"] <= 5, spec)
            self.assertTrue(-5 <= spec["y_pos"] <= 5, spec)


if __name__ == "__main__":
    unittest.main()
