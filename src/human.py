"""Human agent: ranged attacker with a weapon tier."""

from __future__ import annotations

import random
from dataclasses import dataclass

from src.agent import Agent
from utils import combat

# Stop a hair inside weapon range: parking on the exact boundary leaves a
# float residue of ~1e-14 outside it, and ``can_reach`` is a <= test, so the
# human would never fire (which is the old no-progress stall in disguise).
_RANGE_EPS = 1e-6


@dataclass
class Human(Agent):
    weapon_tier: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.weapon_tier = max(1, int(self.weapon_tier))

    @property
    def max_range(self) -> float:
        return combat.human_max_range(self.weapon_tier)

    @property
    def hold_distance(self) -> float:
        """Stand at the edge of weapon range and shoot from there."""
        return max(0.0, self.max_range - _RANGE_EPS)

    def can_reach(self, distance: float) -> bool:
        return distance <= self.max_range

    def hit_chance(self, distance: float) -> float:
        return combat.human_hit_chance(self.weapon_tier, distance)

    def roll_damage(self, rng: random.Random) -> int:
        return combat.human_damage(self.weapon_tier, rng)

    def special(self) -> int:
        return self.weapon_tier
