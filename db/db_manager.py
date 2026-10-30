"""Read/write agents between PostgreSQL and the Python class system."""

from __future__ import annotations

import psycopg

from config.db_config import connect_params
from config.simulation_config import MAP_MAX, MAP_MIN
from src.human import Human
from src.zombie import Zombie


# Rows per bulk UPDATE statement. Five parameters per row keeps us far below
# PostgreSQL's 65535-parameter limit.
_BULK_CHUNK = 1000


def _clamp_int(value: float) -> int:
    return max(MAP_MIN, min(MAP_MAX, int(round(value))))


class DBManager:
    """Thin data-access layer used by the simulation."""

    def __init__(self, config: dict):
        self.config = config
        self.conn = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> "DBManager":
        self.conn = psycopg.connect(**connect_params(self.config))
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "DBManager":
        return self.connect()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def insert_agents(self, specs: list[dict]) -> list[tuple[int, str]]:
        """Insert agents into the supertable + matching subtable.

        ``specs`` entries carry the common fields plus ``weapon_tier``
        (humans) or ``strength`` (zombies). Returns ``(agent_id, type)``.
        """
        created: list[tuple[int, str]] = []
        assert self.conn is not None
        try:
            with self.conn.cursor() as cur:
                for spec in specs:
                    cur.execute(
                        """
                        INSERT INTO agents
                            (agent_type, is_alive, x_pos, y_pos, health, speed)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING agent_id
                        """,
                        (
                            spec["agent_type"],
                            True,
                            _clamp_int(spec["x_pos"]),
                            _clamp_int(spec["y_pos"]),
                            int(spec["health"]),
                            int(spec["speed"]),
                        ),
                    )
                    agent_id = cur.fetchone()[0]

                    if spec["agent_type"] == "human":
                        cur.execute(
                            """
                            INSERT INTO humans (agent_id, agent_type, weapon_tier)
                            VALUES (%s, 'human', %s)
                            """,
                            (agent_id, int(spec["weapon_tier"])),
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO zombies (agent_id, agent_type, strength)
                            VALUES (%s, 'zombie', %s)
                            """,
                            (agent_id, int(spec["strength"])),
                        )

                    created.append((agent_id, spec["agent_type"]))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return created

    def save_agents(self, agents) -> int:
        """Persist the mutable agent state (position, health, alive flag).

        One ``UPDATE ... FROM (VALUES ...)`` per chunk instead of one statement
        per agent. ``executemany`` costs a server round trip per row, which is
        by far the largest database cost of a run (~0.3 ms each, worse over a
        network); this turns a 100-agent sector save into a single round trip.
        """
        if not agents:
            return 0
        rows = [
            (
                int(agent.agent_id),
                bool(agent.is_alive),
                _clamp_int(agent.x_pos),
                _clamp_int(agent.y_pos),
                max(0, int(agent.health)),
            )
            for agent in agents
        ]
        assert self.conn is not None
        try:
            with self.conn.cursor() as cur:
                for start in range(0, len(rows), _BULK_CHUNK):
                    chunk = rows[start : start + _BULK_CHUNK]
                    placeholders = ", ".join(["(%s, %s, %s, %s, %s)"] * len(chunk))
                    params = [value for row in chunk for value in row]
                    cur.execute(
                        f"""
                        UPDATE agents AS a
                           SET is_alive = v.is_alive,
                               x_pos = v.x_pos,
                               y_pos = v.y_pos,
                               health = v.health
                          FROM (VALUES {placeholders})
                               AS v(agent_id, is_alive, x_pos, y_pos, health)
                         WHERE a.agent_id = v.agent_id
                        """,
                        params,
                    )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return len(rows)

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def fetch_alive_agents(self) -> list[tuple[int, str, int, int]]:
        """Return ``(agent_id, agent_type, x_pos, y_pos)`` for living agents."""
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_id, agent_type, x_pos, y_pos
                  FROM agents
                 WHERE is_alive
                 ORDER BY agent_id
                """
            )
            rows = cur.fetchall()
        # End the read transaction immediately: worker connections write
        # between rounds, and an idle-in-transaction snapshot would otherwise
        # hold back vacuum for the whole run.
        self.conn.commit()
        return rows

    def load_agents(self, agent_ids: list[int]) -> tuple[list[Human], list[Zombie]]:
        """Hydrate agents from the database into Python objects.

        Only living agents are loaded; dead ones stay in PostgreSQL untouched.
        """
        if not agent_ids:
            return [], []

        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.agent_id, a.agent_type, a.is_alive, a.x_pos, a.y_pos,
                       a.health, a.speed, h.weapon_tier, z.strength
                  FROM agents a
                  LEFT JOIN humans h
                         ON h.agent_id = a.agent_id
                        AND h.agent_type = a.agent_type
                  LEFT JOIN zombies z
                         ON z.agent_id = a.agent_id
                        AND z.agent_type = a.agent_type
                 WHERE a.agent_id = ANY(%s)
                   AND a.is_alive
                 ORDER BY a.agent_id
                """,
                (list(agent_ids),),
            )
            rows = cur.fetchall()

        humans: list[Human] = []
        zombies: list[Zombie] = []
        for (
            agent_id,
            agent_type,
            is_alive,
            x_pos,
            y_pos,
            health,
            speed,
            weapon_tier,
            strength,
        ) in rows:
            if agent_type == "human":
                humans.append(
                    Human(
                        agent_id=agent_id,
                        agent_type="human",
                        is_alive=is_alive,
                        x_pos=float(x_pos),
                        y_pos=float(y_pos),
                        health=int(health),
                        speed=int(speed),
                        weapon_tier=int(weapon_tier or 1),
                    )
                )
            else:
                zombies.append(
                    Zombie(
                        agent_id=agent_id,
                        agent_type="zombie",
                        is_alive=is_alive,
                        x_pos=float(x_pos),
                        y_pos=float(y_pos),
                        health=int(health),
                        speed=int(speed),
                        strength=int(strength or 1),
                    )
                )
        return humans, zombies

    # ------------------------------------------------------------------ #
    # Stats
    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        """Counts used by the status screen and the end-of-run report."""
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_type,
                       count(*) FILTER (WHERE is_alive)     AS alive,
                       count(*) FILTER (WHERE NOT is_alive) AS dead
                  FROM agents
                 GROUP BY agent_type
                """
            )
            rows = cur.fetchall()

        by_type = {row[0]: {"alive": row[1], "dead": row[2]} for row in rows}
        return {
            "humans": by_type.get("human", {"alive": 0, "dead": 0}),
            "zombies": by_type.get("zombie", {"alive": 0, "dead": 0}),
        }
