"""Zombie agent: short fixed-range melee attacker."""

from __future__ import annotations

import random
from dataclasses import dataclass

from config import simulation_config as cfg
from src.agent import Agent
from utils import combat


@dataclass
class Zombie(Agent):
    strength: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.strength = max(1, int(self.strength))

    @property
    def attack_range(self) -> float:
        """Fixed reach - zombies cannot attack from farther away."""
        return cfg.ZOMBIE_ATTACK_RANGE

    @property
    def hold_distance(self) -> float:
        """How close a zombie closes before it stops."""
        return cfg.ZOMBIE_STOP_DISTANCE

    def can_reach(self, distance: float) -> bool:
        return distance <= self.attack_range

    @property
    def is_super(self) -> bool:
        # No DB flag: a strength above the spawn maximum is what marks a super.
        return self.strength > cfg.ZOMBIE_STRENGTH_RANGE[1]

    def roll_damage(self, rng: random.Random) -> int:
        return combat.zombie_damage(self.strength, rng)
