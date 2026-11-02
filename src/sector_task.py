"""The picklable unit of parallel work: one sector visit.

A task carries everything needed to run on a worker of any backend:

* database mode - ``db_config``; the worker loads its own sector, simulates it
  and writes it back on a connection owned by that worker thread/process.  At
  most ``workers`` sectors are resident in memory at once, so the project's
  "never hold the whole world in memory" rule still holds.
* memory mode - ``humans``/``zombies`` payloads; nothing touches PostgreSQL.
  Used by the benchmark and the equivalence tests.

Both modes return the same :class:`~src.sector_engine.SectorOutcome`, so the
orchestrator does not care where a sector ran.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field

from db.db_manager import DBManager
from src.agent import Agent
from src.sector_engine import SectorEngine, SectorOutcome
from utils.sectoring import Sector


@dataclass
class SectorTask:
    sector: Sector
    seed: int
    round_index: int = 1
    sectors_in_round: int = 0
    capture_frames: bool = False
    max_ticks: int | None = None
    event_shards: int = 1
    db_config: dict | None = None
    humans: list[Agent] | None = field(default=None, repr=False)
    zombies: list[Agent] | None = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Per-worker database connections
# --------------------------------------------------------------------------- #
# One connection per worker thread keeps the code identical for both pool
# types: a process naturally has one thread, and a thread pool gets a
# connection per thread (psycopg connections must not be shared).
_STATE = threading.local()
_CONNECTIONS: list[DBManager] = []
_LOCK = threading.Lock()


def worker_db(config: dict) -> DBManager:
    """Return this worker's connection, opening it on first use."""
    db = getattr(_STATE, "db", None)
    if db is None:
        db = DBManager(config).connect()
        _STATE.db = db
        with _LOCK:
            _CONNECTIONS.append(db)
    return db


def close_all_connections() -> None:
    """Close every connection this process opened.

    Called by the pool on shutdown (as a process finalizer, or from the main
    thread after a thread pool has been joined).
    """
    with _LOCK:
        connections, _CONNECTIONS[:] = list(_CONNECTIONS), []
    for db in connections:
        db.close()


# --------------------------------------------------------------------------- #
# Warm-up
# --------------------------------------------------------------------------- #
def warm_worker(db_config: dict | None = None) -> None:
    """Pay the start-up cost in the worker, not in the first sector.

    Importing the engine and opening a PostgreSQL connection is most of what a
    fresh worker does before its first task.  Running this in the background
    while the orchestrator is still writing spawns means the first round starts
    with workers that are already alive and connected.
    """
    if db_config:
        worker_db(db_config)


# --------------------------------------------------------------------------- #
# Task entry point
# --------------------------------------------------------------------------- #
def _final_states(humans, zombies) -> list[tuple]:
    """Compact post-visit state used to avoid a full re-SELECT per round."""
    return [
        (agent.agent_id, agent.agent_type, agent.x_pos, agent.y_pos, agent.is_alive)
        for agent in (*humans.values(), *zombies.values())
    ]


def simulate_sector_task(task: SectorTask) -> SectorOutcome:
    engine = SectorEngine(
        max_ticks=task.max_ticks,
        capture_frames=task.capture_frames,
        event_shards=task.event_shards,
    )
    rng = random.Random(task.seed)

    if task.humans is not None or task.zombies is not None:
        humans = {agent.agent_id: agent for agent in (task.humans or [])}
        zombies = {agent.agent_id: agent for agent in (task.zombies or [])}
        outcome = engine.run(
            task.sector,
            humans,
            zombies,
            rng,
            round_index=task.round_index,
            sectors_in_round=task.sectors_in_round,
        )
        outcome.final_states = _final_states(humans, zombies)
        return outcome

    if not task.db_config:
        raise ValueError("SectorTask needs either agent payloads or a db_config")

    db = worker_db(task.db_config)
    human_list, zombie_list = db.load_agents(task.sector.agent_ids)
    humans = {agent.agent_id: agent for agent in human_list}
    zombies = {agent.agent_id: agent for agent in zombie_list}

    outcome = engine.run(
        task.sector,
        humans,
        zombies,
        rng,
        round_index=task.round_index,
        sectors_in_round=task.sectors_in_round,
    )
    outcome.final_states = _final_states(humans, zombies)

    # Write the visit back before the worker moves on to the next sector, so
    # the orchestrator's next read sees every sector's result.
    db.save_agents([*humans.values(), *zombies.values()])
    return outcome
