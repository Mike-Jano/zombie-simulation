"""Movement rules: humans hold at weapon range, zombies keep closing."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from src.human import Human  # noqa: E402
from src.sector_engine import SectorEngine  # noqa: E402
from src.zombie import Zombie  # noqa: E402

BOUNDS = (-1000.0, -1000.0, 1000.0, 1000.0)


def sector() -> SimpleNamespace:
    return SimpleNamespace(bounds=BOUNDS)


def make_human(x: float = 0.0, speed: int = 10, tier: int = 1) -> Human:
    return Human(
        agent_id=1,
        agent_type="human",
        is_alive=True,
        x_pos=x,
        y_pos=0.0,
        health=100,
        speed=speed,
        weapon_tier=tier,
    )


def make_zombie(x: float, speed: int = 3) -> Zombie:
    return Zombie(
        agent_id=2,
        agent_type="zombie",
        is_alive=True,
        x_pos=x,
        y_pos=0.0,
        health=100,
        speed=speed,
        strength=1,
    )


def move(human: Human, zombie: Zombie):
    humans = {human.agent_id: human}
    zombies = {zombie.agent_id: zombie}
    return SectorEngine._movement_phase(sector(), humans, zombies)


class HumanStandoffTests(unittest.TestCase):
    def test_hold_distance_is_weapon_range(self):
        for tier in (1, 3, 5):
            human = make_human(tier=tier)
            self.assertAlmostEqual(
                human.hold_distance, tier * cfg.HUMAN_RANGE_PER_TIER, places=5
            )
            # Never outside the range it is supposed to fire from.
            self.assertLessEqual(human.hold_distance, human.max_range)

    def test_human_stops_at_range(self):
        # Human at x=0, zombie at x=10. The human moves first, toward the
        # zombie's position at the start of the tick, and must park a hair
        # inside its own weapon range (10 - max_range), not at half range.
        human = make_human()
        zombie = make_zombie(10.0)
        move(human, zombie)
        self.assertAlmostEqual(human.x_pos, 10.0 - human.max_range, places=4)

    def test_human_inside_range_does_not_move(self):
        human = make_human()
        zombie = make_zombie(2.0)
        move(human, zombie)
        self.assertEqual(human.x_pos, 0.0)

    def test_zombie_still_closes(self):
        human = make_human(x=100.0)
        zombie = make_zombie(0.0)
        move(human, zombie)
        self.assertGreater(zombie.x_pos, 0.0)
        self.assertAlmostEqual(zombie.x_pos, 3.0, places=4)

    def test_human_parked_at_range_can_attack(self):
        # Exactly at the boundary the human is inside range and attacks; the
        # zombie's 2.5-unit reach does not. This is the guard for the float
        # residue that would otherwise leave the human unable to fire.
        human = make_human()
        zombie = make_zombie(human.max_range)
        events, attacks = SectorEngine._combat_phase(
            sector(), {human.agent_id: human}, {zombie.agent_id: zombie}, random.Random(0)
        )
        self.assertEqual(attacks, 1)
        self.assertTrue(events[0].startswith("H"), events)


if __name__ == "__main__":
    unittest.main()
