"""Combat tuning helpers shared by the agent classes."""

from __future__ import annotations

import random

from config import simulation_config as cfg


def human_max_range(weapon_tier: int) -> float:
    return weapon_tier * cfg.HUMAN_RANGE_PER_TIER


def human_hit_chance(weapon_tier: int, distance: float) -> float:
    """Higher tiers are more accurate at every distance and reach farther."""
    if distance > human_max_range(weapon_tier):
        return 0.0
    chance = cfg.HUMAN_HIT_BASE - (cfg.HUMAN_HIT_FALLOFF * distance / weapon_tier)
    return max(cfg.HUMAN_HIT_FLOOR, min(cfg.HUMAN_HIT_BASE, chance))


def human_damage(weapon_tier: int, rng: random.Random) -> int:
    low_mult, high_mult = cfg.HUMAN_DAMAGE_PER_TIER
    return rng.randint(weapon_tier * low_mult, weapon_tier * high_mult)


def zombie_damage(strength: int, rng: random.Random) -> int:
    low_mult, high_mult = cfg.ZOMBIE_DAMAGE_PER_STRENGTH
    return rng.randint(strength * low_mult, strength * high_mult)
