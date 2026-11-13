"""Super agents: rare doubled stats, triple health, fixed speed, and the flag."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from src.human import Human  # noqa: E402
from src.spawner import generate_spawns  # noqa: E402
from src.zombie import Zombie  # noqa: E402

BOUNDS = (-10.0, -10.0, 10.0, 10.0)


def spawn(humans: int, zombies: int) -> list[dict]:
    return generate_spawns(humans, zombies, random.Random(7), map_bounds=BOUNDS)


class SuperSpawnTests(unittest.TestCase):
    def test_forced_super_roll_gives_the_doubled_spread(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 1.0):
            specs = spawn(3, 4)

        humans = [spec for spec in specs if spec["agent_type"] == "human"]
        zombies = [spec for spec in specs if spec["agent_type"] == "zombie"]
        self.assertEqual(len(humans), 3)
        self.assertEqual(len(zombies), 4)
        for spec in humans:
            self.assertTrue(spec["is_super"])
            self.assertEqual(spec["health"], cfg.HUMAN_HEALTH_RANGE[1] * 3)
            self.assertEqual(spec["speed"], cfg.SUPER_SPEED)
            self.assertEqual(spec["weapon_tier"], cfg.HUMAN_WEAPON_TIER_RANGE[1] * 2)
        for spec in zombies:
            self.assertTrue(spec["is_super"])
            self.assertEqual(spec["health"], cfg.ZOMBIE_HEALTH_RANGE[1] * 3)
            self.assertEqual(spec["speed"], cfg.SUPER_SPEED)
            self.assertEqual(spec["strength"], cfg.ZOMBIE_STRENGTH_RANGE[1] * 2)

    def test_zero_chance_keeps_every_stat_in_the_normal_range(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.0):
            specs = spawn(20, 20)

        for spec in specs:
            self.assertFalse(spec["is_super"])
            if spec["agent_type"] == "human":
                self.assertIn(spec["health"], range(*_inclusive(cfg.HUMAN_HEALTH_RANGE)))
                self.assertIn(spec["speed"], range(*_inclusive(cfg.HUMAN_SPEED_RANGE)))
                self.assertIn(
                    spec["weapon_tier"],
                    range(*_inclusive(cfg.HUMAN_WEAPON_TIER_RANGE)),
                )
            else:
                self.assertIn(spec["health"], range(*_inclusive(cfg.ZOMBIE_HEALTH_RANGE)))
                self.assertIn(spec["speed"], range(*_inclusive(cfg.ZOMBIE_SPEED_RANGE)))
                self.assertIn(
                    spec["strength"], range(*_inclusive(cfg.ZOMBIE_STRENGTH_RANGE))
                )


def _inclusive(bounds: tuple[int, int]) -> tuple[int, int]:
    return bounds[0], bounds[1] + 1


class SuperFlagTests(unittest.TestCase):
    def test_super_flag_is_stored_not_inferred(self):
        self.assertFalse(Human(1, "human", True, 0.0, 0.0, 100, 3, 5).is_super)
        self.assertTrue(
            Human(2, "human", True, 0.0, 0.0, 450, 3, 10, is_super=True).is_super
        )
        self.assertFalse(Zombie(3, "zombie", True, 0.0, 0.0, 100, 3, 5).is_super)
        self.assertTrue(
            Zombie(4, "zombie", True, 0.0, 0.0, 600, 3, 10, is_super=True).is_super
        )

    def test_snapshot_carries_the_super_flag(self):
        super_human = Human(
            2, "human", True, 0.0, 0.0, 450, 3, 10, is_super=True
        )
        normal_zombie = Zombie(3, "zombie", True, 0.0, 0.0, 100, 3, 5)
        self.assertTrue(super_human.snapshot()[6])
        self.assertFalse(normal_zombie.snapshot()[6])


if __name__ == "__main__":
    unittest.main()
