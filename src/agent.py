"""Agent superclass: state and behaviour shared by humans and zombies."""

from __future__ import annotations

import math
from dataclasses import dataclass

Bounds = tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


@dataclass
class Agent:
    """Mirror of one row of the ``agents`` PostgreSQL table."""

    agent_id: int
    agent_type: str
    is_alive: bool
    x_pos: float
    y_pos: float
    health: int
    speed: int

    def __post_init__(self) -> None:
        self.health = max(0, int(self.health))
        if self.health == 0:
            self.is_alive = False

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    def planned_move(
        self,
        target_x: float,
        target_y: float,
        stop_distance: float = 0.0,
        bounds: Bounds | None = None,
    ) -> tuple[float, float, float]:
        """Compute a move without mutating the agent.

        Returns ``(new_x, new_y, moved)``.  Keeping the arithmetic in one place
        lets the parallel movement phase apply moves in a deterministic order.
        """
        if not self.is_alive:
            return self.x_pos, self.y_pos, 0.0

        dx = target_x - self.x_pos
        dy = target_y - self.y_pos
        distance = math.hypot(dx, dy)
        if distance <= stop_distance or distance == 0.0:
            return self.x_pos, self.y_pos, 0.0

        travel = min(float(self.speed), distance - stop_distance)
        if travel <= 0.0:
            return self.x_pos, self.y_pos, 0.0

        ux, uy = dx / distance, dy / distance
        new_x = self.x_pos + ux * travel
        new_y = self.y_pos + uy * travel

        if bounds is not None:
            min_x, min_y, max_x, max_y = bounds
            new_x = min_max(new_x, min_x, max_x)
            new_y = min_max(new_y, min_y, max_y)

        moved = math.hypot(new_x - self.x_pos, new_y - self.y_pos)
        return new_x, new_y, moved

    # ------------------------------------------------------------------ #
    # Damage
    # ------------------------------------------------------------------ #
    def take_damage(self, damage: int) -> None:
        if not self.is_alive or damage <= 0:
            return
        self.health -= int(damage)
        if self.health <= 0:
            self.health = 0
            self.is_alive = False

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    @property
    def is_super(self) -> bool:
        """Rare over-max agent; subclasses check their own stat against spawn max."""
        return False

    def snapshot(self) -> tuple:
        """Compact tuple consumed by the visualisers (last item is the super flag)."""
        return (
            self.agent_id,
            self.agent_type,
            round(self.x_pos, 1),
            round(self.y_pos, 1),
            int(self.health),
            bool(self.is_alive),
            bool(self.is_super),
        )


def min_max(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
