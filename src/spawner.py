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
) -> tuple[int, int, int, bool]:
    """Roll ``(health, speed, stat, is_super)``.

    Every agent draws the same four values, so whether an agent turns out to be
    super never shifts the stream for the agents spawned after it.
    """
    health = rng.randint(*health_range)
    speed = rng.randint(*speed_range)
    stat = rng.randint(*stat_range)
    is_super = rng.random() < cfg.SUPER_AGENT_CHANCE
    if is_super:
        health = health_range[1] * cfg.SUPER_HEALTH_MULTIPLIER
        speed = cfg.SUPER_SPEED
        stat = stat_range[1] * cfg.SUPER_STAT_MULTIPLIER
    return health, speed, stat, is_super


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
        health, speed, weapon_tier, is_super = _random_stats(
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
                "is_super": is_super,
            }
        )

    for _ in range(max(0, zombies)):
        x, y = _random_position(rng, map_bounds)
        health, speed, strength, is_super = _random_stats(
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
                "is_super": is_super,
            }
        )

    return specs


def promote_champions(
    champions: list[dict],
    rng: random.Random | None = None,
    map_bounds: Bounds | None = None,
) -> list[dict]:
    """Turn the previous run's survivors into upgraded, ready-to-insert specs.

    Draw order per champion (ordered by ``agent_id`` by the caller): speed
    upgrade, combat-stat upgrade, super roll, position. Keeping these draws
    fixed makes a seeded carry-over run reproducible.
    """
    if not champions:
        return []
    rng = rng or random.Random()
    if map_bounds is None:
        map_bounds = map_bounds_for(len(champions))

    promoted: list[dict] = []
    for champion in champions:
        agent_type = champion["agent_type"]
        human = agent_type == "human"
        max_health = (
            cfg.HUMAN_HEALTH_RANGE[1] if human else cfg.ZOMBIE_HEALTH_RANGE[1]
        )
        max_stat = (
            cfg.HUMAN_WEAPON_TIER_RANGE[1]
            if human
            else cfg.ZOMBIE_STRENGTH_RANGE[1]
        )
        stat_key = "weapon_tier" if human else "strength"

        health = max_health
        speed = int(champion["speed"]) + rng.randint(
            *cfg.CHAMPION_STAT_UPGRADE_RANGE
        )
        stat = int(champion[stat_key] or 1) + rng.randint(
            *cfg.CHAMPION_STAT_UPGRADE_RANGE
        )
        is_super_roll = rng.random() < cfg.SUPER_AGENT_CHANCE
        is_super = bool(champion.get("is_super")) or is_super_roll
        if is_super:
            health = max_health * cfg.SUPER_HEALTH_MULTIPLIER
            speed = cfg.SUPER_SPEED
            stat = max_stat * cfg.SUPER_STAT_MULTIPLIER

        x, y = _random_position(rng, map_bounds)
        promoted.append(
            {
                "agent_id": int(champion["agent_id"]),
                "agent_type": agent_type,
                "x_pos": x,
                "y_pos": y,
                "health": health,
                "speed": speed,
                stat_key: stat,
                "is_super": is_super,
            }
        )
    return promoted


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
            "is_super": spec["is_super"],
        }
        if spec["agent_type"] == "human":
            agents.append(Human(**common, weapon_tier=spec["weapon_tier"]))
        else:
            agents.append(Zombie(**common, strength=spec["strength"]))
    return agents
