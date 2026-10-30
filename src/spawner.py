"""Random population generation for a new simulation run."""

from __future__ import annotations

import random

from config import simulation_config as cfg

def _random_position(rng: random.Random) -> tuple[int, int]:
    return (
        rng.randint(cfg.MAP_MIN, cfg.MAP_MAX),
        rng.randint(cfg.MAP_MIN, cfg.MAP_MAX),
    )


def generate_spawns(
    humans: int, zombies: int, rng: random.Random | None = None
) -> list[dict]:
    """Build insert-ready dicts for ``humans`` humans and ``zombies`` zombies."""
    rng = rng or random.Random()

    specs: list[dict] = []
    for _ in range(max(0, humans)):
        x, y = _random_position(rng)
        specs.append(
            {
                "agent_type": "human",
                "x_pos": x,
                "y_pos": y,
                "health": rng.randint(*cfg.HUMAN_HEALTH_RANGE),
                "speed": rng.randint(*cfg.HUMAN_SPEED_RANGE),
                "weapon_tier": rng.randint(*cfg.HUMAN_WEAPON_TIER_RANGE),
            }
        )

    for _ in range(max(0, zombies)):
        x, y = _random_position(rng)
        specs.append(
            {
                "agent_type": "zombie",
                "x_pos": x,
                "y_pos": y,
                "health": rng.randint(*cfg.ZOMBIE_HEALTH_RANGE),
                "speed": rng.randint(*cfg.ZOMBIE_SPEED_RANGE),
                "strength": rng.randint(*cfg.ZOMBIE_STRENGTH_RANGE),
            }
        )

    return specs
