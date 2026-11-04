"""Random population generation for a new simulation run."""

from __future__ import annotations

import random

from config import simulation_config as cfg
from src.human import Human
from src.zombie import Zombie
from utils.sectoring import Bounds, map_bounds_for


def _random_position(rng: random.Random, bounds: Bounds) -> tuple[int, int]:
    min_x, min_y, max_x, max_y = bounds
    return (
        rng.randint(int(min_x), int(max_x)),
        rng.randint(int(min_y), int(max_y)),
    )


def generate_spawns(
    humans: int,
    zombies: int,
    rng: random.Random | None = None,
    map_bounds: Bounds | None = None,
) -> list[dict]:
    """Build insert-ready dicts for ``humans`` humans and ``zombies`` zombies."""
    rng = rng or random.Random()
    if map_bounds is None:
        map_bounds = map_bounds_for(max(0, humans) + max(0, zombies))

    specs: list[dict] = []
    for _ in range(max(0, humans)):
        x, y = _random_position(rng, map_bounds)
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
        x, y = _random_position(rng, map_bounds)
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


def build_agents(
    humans: int, zombies: int, rng: random.Random | None = None
) -> list[Human | Zombie]:
    """Build in-memory agents from :func:`generate_spawns` (tests, benchmark)."""
    rng = rng or random.Random()
    agents: list[Human | Zombie] = []
    for index, spec in enumerate(generate_spawns(humans, zombies, rng), start=1):
        common = {
            "agent_id": index,
            "agent_type": spec["agent_type"],
            "is_alive": True,
            "x_pos": spec["x_pos"],
            "y_pos": spec["y_pos"],
            "health": spec["health"],
            "speed": spec["speed"],
        }
        if spec["agent_type"] == "human":
            agents.append(Human(**common, weapon_tier=spec["weapon_tier"]))
        else:
            agents.append(Zombie(**common, strength=spec["strength"]))
    return agents
