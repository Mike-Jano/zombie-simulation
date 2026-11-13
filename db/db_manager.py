"""Read/write agents between PostgreSQL and the Python class system."""

from __future__ import annotations

import psycopg

from config.db_config import connect_params
from src.human import Human
from src.zombie import Zombie


# Rows per bulk statement. Keeps us far below PostgreSQL's 65535-parameter limit.
_BULK_CHUNK = 1000

# Session-level advisory lock so two simulations cannot share one database.
RUN_LOCK_KEY = 0x5A4D4249  # "ZMBI"

Bounds = tuple[float, float, float, float]


def _clamp_int(value: float, bounds: Bounds | None = None) -> int:
    rounded = int(round(value))
    if bounds is None:
        # No run bounds supplied (status/reset/tests): store the rounded value.
        return rounded
    return max(int(bounds[0]), min(int(bounds[2]), rounded))


def _chunks(rows: list, size: int = _BULK_CHUNK):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


class DBManager:
    """Thin data-access layer used by the simulation."""

    def __init__(self, config: dict, map_bounds: Bounds | None = None):
        self.config = config
        self.map_bounds = map_bounds
        self.conn = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> "DBManager":
        self.conn = psycopg.connect(**connect_params(self.config))
        return self

    def close(self) -> None:
        if self.conn is not None:
            # Closing the session also releases the advisory run lock.
            self.conn.close()
            self.conn = None

    def try_run_lock(self) -> bool:
        """Claim this database for one run.

        The lock is session-level, so it is released by :meth:`close` (or when
        the process dies). A second run gets ``False`` instead of truncating
        the first run's agents.
        """
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (RUN_LOCK_KEY,))
            acquired = bool(cur.fetchone()[0])
        self.conn.commit()
        return acquired

    def __enter__(self) -> "DBManager":
        return self.connect()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def insert_agents(self, specs: list[dict]) -> None:
        """Insert agents into the supertable and the matching subtable.

        ``specs`` entries carry the common fields plus ``weapon_tier``
        (humans) or ``strength`` (zombies). Rows are inserted in bulk so a
        large spawn is a handful of statements, not two per agent.
        """
        if not specs:
            return
        assert self.conn is not None
        try:
            with self.conn.cursor() as cur:
                # RETURNING row order is not guaranteed for a multi-row INSERT,
                # so allocate the IDs first and pair them with the specs here.
                cur.execute(
                    "SELECT pg_get_serial_sequence('agents', 'agent_id')"
                )
                sequence = cur.fetchone()[0]
                if sequence is None:
                    raise RuntimeError("agents.agent_id has no sequence")

                agent_ids: list[int] = []
                for chunk in _chunks(specs):
                    cur.execute(
                        "SELECT nextval(%s::regclass) FROM generate_series(1, %s)",
                        (sequence, len(chunk)),
                    )
                    ids = [row[0] for row in cur.fetchall()]
                    if len(ids) != len(chunk):
                        raise RuntimeError(
                            "agent id allocation returned an unexpected row count"
                        )

                    placeholders = ", ".join(
                        ["(%s, %s, %s, %s, %s, %s, %s, %s)"] * len(chunk)
                    )
                    params = []
                    for agent_id, spec in zip(ids, chunk):
                        params.extend(
                            (
                                agent_id,
                                spec["agent_type"],
                                True,
                                _clamp_int(spec["x_pos"], self.map_bounds),
                                _clamp_int(spec["y_pos"], self.map_bounds),
                                int(spec["health"]),
                                int(spec["speed"]),
                                bool(spec.get("is_super", False)),
                            )
                        )
                    cur.execute(
                        f"""
                        INSERT INTO agents
                            (agent_id, agent_type, is_alive, x_pos, y_pos,
                             health, speed, is_super)
                        VALUES {placeholders}
                        """,
                        params,
                    )
                    agent_ids.extend(ids)

                if len(agent_ids) != len(specs):
                    raise RuntimeError(
                        "agent insert returned an unexpected row count"
                    )

                humans = [
                    (agent_id, int(spec["weapon_tier"]))
                    for agent_id, spec in zip(agent_ids, specs)
                    if spec["agent_type"] == "human"
                ]
                zombies = [
                    (agent_id, int(spec["strength"]))
                    for agent_id, spec in zip(agent_ids, specs)
                    if spec["agent_type"] == "zombie"
                ]
                for table, column, rows in (
                    ("humans", "weapon_tier", humans),
                    ("zombies", "strength", zombies),
                ):
                    for chunk in _chunks(rows):
                        placeholders = ", ".join(["(%s, %s)"] * len(chunk))
                        cur.execute(
                            f"""
                            INSERT INTO {table} (agent_id, {column})
                            VALUES {placeholders}
                            """,
                            [value for row in chunk for value in row],
                        )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

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
                _clamp_int(agent.x_pos, self.map_bounds),
                _clamp_int(agent.y_pos, self.map_bounds),
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

    def save_champion_upgrades(self, rows: list[dict]) -> int:
        """Write promoted champion stats and positions back in place."""
        if not rows:
            return 0
        assert self.conn is not None
        try:
            with self.conn.cursor() as cur:
                for chunk in _chunks(rows):
                    placeholders = ", ".join(
                        ["(%s, %s, %s, %s, %s, %s, %s)"] * len(chunk)
                    )
                    params = []
                    for row in chunk:
                        params.extend(
                            (
                                int(row["agent_id"]),
                                int(row["health"]),
                                int(row["speed"]),
                                _clamp_int(row["x_pos"], self.map_bounds),
                                _clamp_int(row["y_pos"], self.map_bounds),
                                bool(row["is_super"]),
                                True,
                            )
                        )
                    cur.execute(
                        f"""
                        UPDATE agents AS a
                           SET health = v.health,
                               speed = v.speed,
                               x_pos = v.x_pos,
                               y_pos = v.y_pos,
                               is_super = v.is_super,
                               is_champion = v.is_champion
                          FROM (VALUES {placeholders})
                               AS v(agent_id, health, speed, x_pos, y_pos,
                                    is_super, is_champion)
                         WHERE a.agent_id = v.agent_id
                        """,
                        params,
                    )

                for table, column, key in (
                    ("humans", "weapon_tier", "weapon_tier"),
                    ("zombies", "strength", "strength"),
                ):
                    rows_for_table = [
                        (int(row["agent_id"]), int(row[key]))
                        for row in rows
                        if row["agent_type"] == table[:-1]
                    ]
                    for chunk in _chunks(rows_for_table):
                        placeholders = ", ".join(["(%s, %s)"] * len(chunk))
                        cur.execute(
                            f"""
                            UPDATE {table} AS t
                               SET {column} = v.stat_value
                              FROM (VALUES {placeholders})
                                   AS v(agent_id, stat_value)
                             WHERE t.agent_id = v.agent_id
                            """,
                            [value for row in chunk for value in row],
                        )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return len(rows)

    def mark_champions(self) -> int:
        """Flag the run's survivors for the next carry-over."""
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute("UPDATE agents SET is_champion = is_alive")
            updated = cur.rowcount
        self.conn.commit()
        return updated

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #
    def load_champions(self) -> list[dict]:
        """Return the flagged survivors of the previous run."""
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT a.agent_id, a.agent_type, a.health, a.speed, a.is_super,
                       h.weapon_tier, z.strength
                  FROM agents a
                  LEFT JOIN humans h
                         ON h.agent_id = a.agent_id
                        AND h.agent_type = a.agent_type
                  LEFT JOIN zombies z
                         ON z.agent_id = a.agent_id
                        AND z.agent_type = a.agent_type
                 WHERE a.is_champion AND a.is_alive
                 ORDER BY a.agent_id
                """
            )
            rows = cur.fetchall()
        self.conn.commit()
        return [
            {
                "agent_id": row[0],
                "agent_type": row[1],
                "health": row[2],
                "speed": row[3],
                "is_super": row[4],
                "weapon_tier": row[5],
                "strength": row[6],
            }
            for row in rows
        ]

    def count_champions(self) -> dict[str, int]:
        """Count the flagged survivors by side; cheaper than loading rows."""
        assert self.conn is not None
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT agent_type, count(*)
                  FROM agents
                 WHERE is_champion AND is_alive
                 GROUP BY agent_type
                """
            )
            counts = dict(cur.fetchall())
        self.conn.commit()
        return counts

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
                       a.health, a.speed, a.is_super, a.is_champion,
                       h.weapon_tier, z.strength
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
        # A read-only SELECT would otherwise hold its snapshot for the whole
        # sector visit; commit now so the connection is not idle in transaction.
        self.conn.commit()

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
            is_super,
            is_champion,
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
                        is_super=is_super,
                        is_champion=is_champion,
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
                        is_super=is_super,
                        is_champion=is_champion,
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
