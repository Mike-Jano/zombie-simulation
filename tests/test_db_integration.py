"""Real-PostgreSQL checks for the data layer.

Skipped by default. Run with a reachable server and a throwaway database:

    $env:DB_TEST = "1"
    .\\venv\\Scripts\\python.exe -m unittest tests.test_db_integration -v

The test always uses ``zombie_sim_test`` (never the run database) and drops
and recreates its tables in ``setUp``.
"""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.db_config import connect_params, load_db_config  # noqa: E402
from db import schema  # noqa: E402
from db.db_manager import DBManager  # noqa: E402
from src.spawner import generate_spawns  # noqa: E402

TEST_CONFIG = {**load_db_config(), "dbname": "zombie_sim_test"}


@unittest.skipUnless(
    os.environ.get("DB_TEST") == "1",
    "set DB_TEST=1 with a running PostgreSQL server to run this test",
)
class DatabaseIntegrationTests(unittest.TestCase):
    def setUp(self):
        schema.reset_database(TEST_CONFIG)

    def _insert(self, humans: int = 3, zombies: int = 2, seed: int = 7):
        specs = generate_spawns(humans, zombies, random.Random(seed))
        with DBManager(TEST_CONFIG) as db:
            db.insert_agents(specs)
        return specs

    def test_run_lock_is_exclusive(self):
        first = DBManager(TEST_CONFIG).connect()
        second = DBManager(TEST_CONFIG).connect()
        try:
            self.assertTrue(first.try_run_lock())
            self.assertFalse(second.try_run_lock())
        finally:
            first.close()
        try:
            self.assertTrue(second.try_run_lock())
        finally:
            second.close()

    def test_bulk_insert_load_and_stats(self):
        specs = self._insert(3, 2)
        with DBManager(TEST_CONFIG) as db:
            stats = db.stats()
            self.assertEqual(stats["humans"]["alive"], 3)
            self.assertEqual(stats["zombies"]["alive"], 2)

            rows = db.fetch_alive_agents()
            humans, zombies = db.load_agents([row[0] for row in rows])

        self.assertEqual(len(humans), 3)
        self.assertEqual(len(zombies), 2)
        self.assertEqual(
            sorted(agent.agent_id for agent in humans),
            sorted(
                index + 1
                for index, spec in enumerate(specs)
                if spec["agent_type"] == "human"
            ),
        )

    def test_bulk_insert_pairs_each_subtable_row(self):
        specs = [
            {
                "agent_type": "human",
                "x_pos": 11,
                "y_pos": 12,
                "health": 100,
                "speed": 2,
                "weapon_tier": 5,
            },
            {
                "agent_type": "zombie",
                "x_pos": 21,
                "y_pos": 22,
                "health": 120,
                "speed": 3,
                "strength": 4,
            },
            {
                "agent_type": "human",
                "x_pos": 31,
                "y_pos": 32,
                "health": 110,
                "speed": 4,
                "weapon_tier": 2,
            },
        ]
        with DBManager(TEST_CONFIG) as db:
            db.insert_agents(specs)

        with psycopg.connect(**connect_params(TEST_CONFIG)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.agent_type, a.x_pos, a.y_pos, a.health, a.speed,
                           h.weapon_tier, z.strength
                      FROM agents a
                      LEFT JOIN humans h
                             ON h.agent_id = a.agent_id
                            AND h.agent_type = a.agent_type
                      LEFT JOIN zombies z
                             ON z.agent_id = a.agent_id
                            AND z.agent_type = a.agent_type
                    """
                )
                rows = cur.fetchall()

        by_key = {
            (kind, x, y, health, speed): (weapon, strength)
            for kind, x, y, health, speed, weapon, strength in rows
        }
        for spec in specs:
            key = (
                spec["agent_type"],
                spec["x_pos"],
                spec["y_pos"],
                spec["health"],
                spec["speed"],
            )
            weapon, strength = by_key[key]
            if spec["agent_type"] == "human":
                self.assertEqual(weapon, spec["weapon_tier"])
                self.assertIsNone(strength)
            else:
                self.assertIsNone(weapon)
                self.assertEqual(strength, spec["strength"])

    def test_bootstrap_reports_missing_columns_before_index_repair(self):
        self._insert(1, 1)
        with psycopg.connect(**connect_params(TEST_CONFIG)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE agents DROP COLUMN is_alive CASCADE")

        missing = schema.bootstrap(TEST_CONFIG)
        self.assertIn("agents.is_alive", missing)

    def test_bootstrap_recreates_a_missing_subtable(self):
        with psycopg.connect(**connect_params(TEST_CONFIG)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("DROP TABLE humans CASCADE")

        self.assertIsNone(schema.bootstrap(TEST_CONFIG))

        with psycopg.connect(**connect_params(TEST_CONFIG)) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('humans') IS NOT NULL")
                self.assertTrue(cur.fetchone()[0])

    def test_load_ends_its_read_transaction(self):
        from psycopg.pq import TransactionStatus

        self._insert(2, 2)
        with DBManager(TEST_CONFIG) as db:
            rows = db.fetch_alive_agents()
            db.load_agents([row[0] for row in rows])
            self.assertEqual(db.conn.info.transaction_status, TransactionStatus.IDLE)

    def test_save_writes_position_health_and_alive(self):
        self._insert(2, 2)
        with DBManager(TEST_CONFIG) as db:
            rows = db.fetch_alive_agents()
            humans, zombies = db.load_agents([row[0] for row in rows])
            victim = humans[0]
            victim.health = 0
            victim.is_alive = False
            survivor = humans[1]
            survivor.x_pos = -321.0
            survivor.y_pos = 654.0
            db.save_agents([*humans, *zombies])

            stored = db.stats()
            self.assertEqual(stored["humans"]["alive"], 1)
            self.assertEqual(stored["humans"]["dead"], 1)
            alive, _ = db.load_agents([survivor.agent_id])
            self.assertEqual((alive[0].x_pos, alive[0].y_pos), (-321.0, 654.0))

    def test_bootstrap_repairs_indexes_and_clear_empties(self):
        self._insert(1, 1)
        with psycopg.connect(**connect_params(TEST_CONFIG)) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("DROP INDEX IF EXISTS idx_agents_alive")
                cur.execute(
                    "CREATE INDEX idx_agents_position ON agents (x_pos, y_pos)"
                )

        self.assertIsNone(schema.bootstrap(TEST_CONFIG))

        with psycopg.connect(**connect_params(TEST_CONFIG)) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT to_regclass('public.idx_agents_alive') IS NOT NULL, "
                    "to_regclass('public.idx_agents_position')"
                )
                alive_exists, position_exists = cur.fetchone()
        self.assertTrue(alive_exists)
        self.assertIsNone(position_exists)
        self.assertEqual(schema.clear_agents(TEST_CONFIG), 2)


if __name__ == "__main__":
    unittest.main()
