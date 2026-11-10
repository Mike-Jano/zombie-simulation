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


def _random_stats(
    rng: random.Random,
    health_range: tuple[int, int],
    speed_range: tuple[int, int],
    stat_range: tuple[int, int],
) -> tuple[int, int, int]:
    """Roll ``(health, speed, stat)``; rare agents get the super spread.

    Every agent draws the same four values, so whether an agent turns out to be
    super never shifts the stream for the agents spawned after it.
    """
    health = rng.randint(*health_range)
    speed = rng.randint(*speed_range)
    stat = rng.randint(*stat_range)
    if rng.random() < cfg.SUPER_AGENT_CHANCE:
        health = health_range[1] * cfg.SUPER_HEALTH_MULTIPLIER
        speed = cfg.SUPER_SPEED
        stat = stat_range[1] * cfg.SUPER_STAT_MULTIPLIER
    return health, speed, stat


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
        health, speed, weapon_tier = _random_stats(
            rng,
            cfg.HUMAN_HEALTH_RANGE,
            cfg.HUMAN_SPEED_RANGE,
            cfg.HUMAN_WEAPON_TIER_RANGE,
        )
        specs.append(
            {
                "agent_type": "human",
                "x_pos": x,
                "y_pos": y,
                "health": health,
                "speed": speed,
                "weapon_tier": weapon_tier,
            }
        )

    for _ in range(max(0, zombies)):
        x, y = _random_position(rng, map_bounds)
        health, speed, strength = _random_stats(
            rng,
            cfg.ZOMBIE_HEALTH_RANGE,
            cfg.ZOMBIE_SPEED_RANGE,
            cfg.ZOMBIE_STRENGTH_RANGE,
        )
        specs.append(
            {
                "agent_type": "zombie",
                "x_pos": x,
                "y_pos": y,
                "health": health,
                "speed": speed,
                "strength": strength,
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
