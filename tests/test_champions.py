"""Champions: survivor promotion, stat upgrades and the super interaction."""

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
from src.spawner import promote_champions  # noqa: E402

BOUNDS = (-100.0, -100.0, 100.0, 100.0)


def champion(
    agent_id: int,
    agent_type: str,
    *,
    speed: int = 3,
    stat: int = 3,
    is_super: bool = False,
) -> dict:
    row = {
        "agent_id": agent_id,
        "agent_type": agent_type,
        "health": 10,
        "speed": speed,
        "is_super": is_super,
        "weapon_tier": stat if agent_type == "human" else None,
        "strength": stat if agent_type == "zombie" else None,
    }
    return row


class ChampionPromotionTests(unittest.TestCase):
    def test_health_is_reset_to_the_normal_maximum(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.0):
            promoted = promote_champions(
                [champion(1, "human"), champion(2, "zombie")],
                random.Random(1),
                BOUNDS,
            )

        self.assertEqual(promoted[0]["health"], cfg.HUMAN_HEALTH_RANGE[1])
        self.assertEqual(promoted[1]["health"], cfg.ZOMBIE_HEALTH_RANGE[1])

    def test_speed_and_combat_stat_upgrade_by_one_to_three(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.0):
            promoted = promote_champions(
                [champion(1, "human", speed=4, stat=5)],
                random.Random(2),
                BOUNDS,
            )

        row = promoted[0]
        self.assertTrue(1 <= row["speed"] - 4 <= 3)
        self.assertTrue(1 <= row["weapon_tier"] - 5 <= 3)
        self.assertFalse(row["is_super"])

    def test_upgrades_can_exceed_the_normal_spawn_range(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.0):
            promoted = promote_champions(
                [champion(1, "zombie", stat=cfg.ZOMBIE_STRENGTH_RANGE[1])],
                random.Random(3),
                BOUNDS,
            )

        self.assertGreater(promoted[0]["strength"], cfg.ZOMBIE_STRENGTH_RANGE[1])

    def test_forced_super_uses_the_normal_super_spread(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 1.0):
            promoted = promote_champions(
                [champion(1, "human", speed=8), champion(2, "zombie", speed=8)],
                random.Random(4),
                BOUNDS,
            )

        human, zombie = promoted
        self.assertTrue(human["is_super"])
        self.assertEqual(human["health"], cfg.HUMAN_HEALTH_RANGE[1] * 3)
        self.assertEqual(human["speed"], cfg.SUPER_SPEED)
        self.assertEqual(
            human["weapon_tier"], cfg.HUMAN_WEAPON_TIER_RANGE[1] * 2
        )
        self.assertEqual(zombie["health"], cfg.ZOMBIE_HEALTH_RANGE[1] * 3)
        self.assertEqual(zombie["speed"], cfg.SUPER_SPEED)
        self.assertEqual(
            zombie["strength"], cfg.ZOMBIE_STRENGTH_RANGE[1] * 2
        )

    def test_an_existing_super_champion_stays_super_and_still_draws_the_roll(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.0):
            super_champion = promote_champions(
                [champion(1, "zombie", is_super=True)],
                random.Random(5),
                BOUNDS,
            )
            plain_champion = promote_champions(
                [champion(1, "zombie", is_super=False)],
                random.Random(5),
                BOUNDS,
            )

        self.assertTrue(super_champion[0]["is_super"])
        self.assertEqual(
            super_champion[0]["health"], cfg.ZOMBIE_HEALTH_RANGE[1] * 3
        )
        # Same seed and same stats: the skipped roll would shift the position
        # draw, so equal positions prove the roll still happens.
        self.assertEqual(
            (super_champion[0]["x_pos"], super_champion[0]["y_pos"]),
            (plain_champion[0]["x_pos"], plain_champion[0]["y_pos"]),
        )

    def test_same_seed_and_rows_produce_the_same_promotion(self):
        rows = [champion(1, "human"), champion(2, "zombie")]
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.25):
            first = promote_champions(rows, random.Random(9), BOUNDS)
            second = promote_champions(rows, random.Random(9), BOUNDS)

        self.assertEqual(first, second)

    def test_positions_stay_inside_the_map_bounds(self):
        with patch.object(cfg, "SUPER_AGENT_CHANCE", 0.0):
            promoted = promote_champions(
                [champion(index, "human") for index in range(1, 21)],
                random.Random(6),
                BOUNDS,
            )

        for row in promoted:
            self.assertGreaterEqual(row["x_pos"], BOUNDS[0])
            self.assertLessEqual(row["x_pos"], BOUNDS[2])
            self.assertGreaterEqual(row["y_pos"], BOUNDS[1])
            self.assertLessEqual(row["y_pos"], BOUNDS[3])

    def test_no_champions_is_a_no_op(self):
        self.assertEqual(promote_champions([], random.Random(1), BOUNDS), [])


if __name__ == "__main__":
    unittest.main()
