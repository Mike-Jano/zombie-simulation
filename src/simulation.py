"""The simulation engine: cluster the map, simulate sectors, repeat.

Sectors are independent - k-means assigns every agent to exactly one sector and
a visit loads, simulates and unloads before the next one starts - so the
sectors of a round can run concurrently. ``runtime.sector_pool`` picks the
widest backend the machine can actually use:

* **process** on ordinary CPython, where the GIL would serialise threads,
* **thread** on free-threaded builds (and for DB-bound profiling),
* **serial** when only one CPU is available or a pool cannot be started.

How *many* sectors exist is decided by the population, not by the core count
(``utils.sectoring.resolve_sector_count``): 200 agents are one arena, 1500 are
three. A finer split would leave sectors containing only humans or only
zombies, which cannot fight, and the run would end with both sides alive.
Workers only execute the sectors the population produced.

Every sector holds both types, and if a whole round still manages no attack and
no movement, the run collapses to a single global sector instead of stopping -
the two guarantees that a run reaches "one side eliminated".

Each sector visit draws from its own seed derived from ``(round seed, sector
index)``, so a run is reproducible no matter how many workers execute it.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from config import simulation_config as cfg
from db.db_manager import DBManager
from runtime.capabilities import ParallelPlan, resolve_plan
from runtime.sector_pool import POOL_FAILURES, SectorPool
from src.sector_engine import SectorEngine, SectorOutcome, SectorReport, derive_seed
from src.sector_task import SectorTask
from utils.sectoring import build_sectors, resolve_sector_count


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass
class SimulationReport:
    sectors: list[SectorReport] = field(default_factory=list)
    duration: float = 0.0
    rounds: int = 0
    stop_reason: str = ""
    workers: int = 1
    backend: str = "serial"
    cpu_count: int = 1
    sector_count: int = 0
    seed: int | None = None
    collapsed: bool = False

    def _grouped(self) -> dict[int, list[SectorReport]]:
        grouped: dict[int, list[SectorReport]] = {}
        for sector in self.sectors:
            grouped.setdefault(sector.round_index, []).append(sector)
        return grouped

    def _edge_round(self, first: bool) -> list[SectorReport]:
        """Sector visits of the earliest (or latest) refresh round."""
        grouped = self._grouped()
        if not grouped:
            return []
        return grouped[min(grouped) if first else max(grouped)]

    @property
    def humans_start(self) -> int:
        return sum(s.humans_start for s in self._edge_round(True))

    @property
    def zombies_start(self) -> int:
        return sum(s.zombies_start for s in self._edge_round(True))

    @property
    def humans_end(self) -> int:
        return sum(s.humans_end for s in self._edge_round(False))

    @property
    def zombies_end(self) -> int:
        return sum(s.zombies_end for s in self._edge_round(False))

    @property
    def iterations(self) -> int:
        """Total sector-ticks executed (work done, not world time)."""
        return sum(s.iterations for s in self.sectors)

    @property
    def attacks(self) -> int:
        return sum(s.attacks for s in self.sectors)

    def render(self) -> str:
        lines = ["", "=" * 66, "SIMULATION REPORT", "=" * 66]
        grouped = self._grouped()
        single_round = len(grouped) <= 1

        keys = sorted(grouped)
        if len(keys) > 24:
            keys = keys[:8] + [-1] + keys[-8:]

        for round_index in keys:
            if round_index == -1:
                lines.append(f"   ... {len(grouped) - 16} more round(s) ...")
                continue
            group = grouped[round_index]
            lines.append(
                f"Round {round_index:<3} "
                f"sectors {len(group):<3} "
                f"ticks {sum(s.iterations for s in group):<5} "
                f"attacks {sum(s.attacks for s in group):<5} "
                f"humans {sum(s.humans_start for s in group)}->"
                f"{sum(s.humans_end for s in group)}  "
                f"zombies {sum(s.zombies_start for s in group)}->"
                f"{sum(s.zombies_end for s in group)}"
            )
            if single_round:
                for sector in group:
                    left, bottom, right, top = sector.bounds
                    lines.append(
                        f"   sector {sector.index:<2} "
                        f"x[{left:>6.0f},{right:>6.0f}] "
                        f"y[{bottom:>6.0f},{top:>6.0f}]  "
                        f"humans {sector.humans_start}->{sector.humans_end}  "
                        f"zombies {sector.zombies_start}->{sector.zombies_end}  "
                        f"iters {sector.iterations:<4} attacks {sector.attacks:<5} "
                        f"({sector.stop_reason})"
                    )

        lines.append("-" * 66)
        lines.append(
            f"Humans : {self.humans_start} -> {self.humans_end} alive   "
            f"({self.humans_start - self.humans_end} lost)"
        )
        lines.append(
            f"Zombies: {self.zombies_start} -> {self.zombies_end} alive   "
            f"({self.zombies_start - self.zombies_end} lost)"
        )
        lines.append(
            f"Rounds: {self.rounds}   Total iterations: {self.iterations}   "
            f"attacks: {self.attacks}   wall time: {self.duration:.2f}s"
        )
        lines.append(
            f"Parallel: {self.backend} backend, {self.workers} worker(s), "
            f"{self.sector_count} sector(s)/round, {self.cpu_count} CPU(s)"
            + (f", seed {self.seed}" if self.seed is not None else "")
        )
        if self.stop_reason:
            lines.append(f"Stopped: {self.stop_reason}")
        if self.collapsed:
            lines.append(
                "Arena: collapsed to one sector (stagnation failsafe fired)"
            )
        lines.append("=" * 66)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class Simulation:
    def __init__(
        self,
        db: DBManager,
        rng: random.Random | None = None,
        visualizers: list | None = None,
        on_sector_complete=None,
        *,
        workers: int | None = None,
        backend: str | None = None,
        sectors: int | None = None,
        seed: int | None = None,
    ):
        self.db = db
        self.rng = rng or random.Random()
        self.visualizers = visualizers or []
        self.on_sector_complete = on_sector_complete
        self.seed = seed

        # ``None`` means "use the config / auto-detect".
        self.requested_workers = workers
        self.requested_backend = backend
        self.requested_sectors = sectors

        self.plan: ParallelPlan | None = None
        self.workers = 1
        self.engine = SectorEngine(capture_frames=bool(self.visualizers))
        self._pool: SectorPool | None = None
        self._parallel: bool | None = None
        self._backend_used = "serial"
        self._collapse = False

        # The in-memory "tables": only the currently loaded sector(s) live here.
        self.humans: dict = {}
        self.zombies: dict = {}
        self.report = SimulationReport()

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self) -> SimulationReport:
        started = time.perf_counter()

        self.plan = resolve_plan(self.requested_workers, self.requested_backend)

        world_ticks = 0
        round_index = 0
        stop_reason = "round limit"

        try:
            while round_index < cfg.MAX_ROUNDS and world_ticks < cfg.MAX_TOTAL_TICKS:
                rows = self.db.fetch_alive_agents()
                kinds = {row[1] for row in rows}
                if not rows:
                    stop_reason = "no agents left"
                    break
                if not {"human", "zombie"} <= kinds:
                    stop_reason = "one side eliminated"
                    break

                # Re-cluster from CURRENT positions every round, so sectors
                # follow the action instead of being frozen where they started.
                positions = {row[0]: (float(row[2]), float(row[3])) for row in rows}
                types = {row[0]: row[1] for row in rows}
                sector_count = resolve_sector_count(
                    len(rows), requested=self.requested_sectors
                )
                if self._collapse:
                    # Stagnation failsafe: one arena, so the two sides cannot
                    # sit in separate regions forever.
                    sector_count = 1
                sectors = build_sectors(
                    positions,
                    rng=self.rng,
                    target_k=sector_count,
                    types=types,
                )

                # The pool is sized to the work: a run that produces one sector
                # (200 agents) stays single-threaded instead of paying for
                # workers that would have nothing to do.
                if self._parallel is None:
                    self._decide_parallel(len(sectors))

                round_index += 1
                # Widest parallel width actually used (the last round of a run
                # has fewer sectors because most agents are already dead).
                self.report.sector_count = max(
                    self.report.sector_count, len(sectors)
                )

                if round_index == 1:
                    self._start_visualizers(len(sectors), len(rows))

                outcomes = self._run_round(
                    sectors, round_index, self.rng.getrandbits(64)
                )
                if outcomes is None:
                    # The pool died mid-round; the next round runs serially.
                    continue

                round_attacks = sum(o.report.attacks for o in outcomes)
                round_moved = sum(o.moved for o in outcomes)
                self._record_round(outcomes, len(sectors), round_index)

                # World time advanced this round is the longest sector visit,
                # NOT the sum over sectors: otherwise adding workers/sectors
                # would silently shorten the run.
                world_ticks += max(o.report.iterations for o in outcomes)

                # A whole round with no fighting and no movement used to end the
                # run with both sides alive. Instead, drop to a single arena and
                # keep going: with both types in one sector every tick lands a
                # hit or closes a distance, so the run can only finish with one
                # side eliminated.
                if round_attacks == 0 and round_moved < 1e-6:
                    action = self._stagnation_action(round_attacks, round_moved)
                    if action == "collapse":
                        self._collapse = True
                        print(
                            f"  [arena] no attacks and no movement in round "
                            f"{round_index} with both sides alive - collapsing "
                            "to one sector"
                        )
                        continue
                    if action == "stop":
                        stop_reason = "no progress"
                        break
            else:
                stop_reason = (
                    "tick budget"
                    if world_ticks >= cfg.MAX_TOTAL_TICKS
                    else "round limit"
                )
        finally:
            self.close()
            for visualizer in self.visualizers:
                visualizer.finish()

        self.report.rounds = round_index
        self.report.stop_reason = stop_reason
        self.report.duration = time.perf_counter() - started
        self.report.workers = self.workers
        self.report.backend = self._active_backend()
        self.report.cpu_count = self.plan.cpu_count if self.plan else 1
        self.report.seed = self.seed
        self.report.collapsed = self._collapse
        return self.report

    # ------------------------------------------------------------------ #
    # Pool plumbing
    # ------------------------------------------------------------------ #
    def _stagnation_action(self, attacks: int, moved: float) -> str:
        """What a round that achieved nothing means.

        ``""`` keep going (something did happen), ``"collapse"`` retry the next
        round as a single arena, ``"stop"`` nothing more can happen.
        """
        if attacks or moved >= 1e-6:
            return ""
        if cfg.COLLAPSE_ON_STAGNATION and not self._collapse:
            return "collapse"
        return "stop"

    def prewarm(self, agent_count: int) -> None:
        """Start the workers early, while the caller is still busy.

        Called by the UI right after spawning, so the ~0.5 s of process
        start-up overlaps with writing the spawns instead of delaying the first
        round.  Safe to skip: :meth:`run` makes the same decision itself.
        """
        if self._parallel is not None or agent_count <= 0:
            return
        self.plan = resolve_plan(self.requested_workers, self.requested_backend)
        sector_count = resolve_sector_count(
            agent_count, requested=self.requested_sectors
        )
        self._decide_parallel(sector_count)
        if self._pool is not None:
            self._pool.warm(self.db.config)

    def close(self) -> None:
        """Release the pool, if it is still open (idempotent)."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _decide_parallel(self, sector_count: int) -> None:
        """Choose once, before the first round, whether to use a pool.

        Width is capped by the sector count, never the other way round: workers
        execute the sectors the population produced, they do not create more.
        """
        plan = self.plan
        if plan is None or not plan.is_parallel or sector_count <= 1:
            self._parallel = False
            self.workers = 1
            return

        self.workers = min(plan.workers, sector_count)
        self._pool = self._open_pool(self.workers)
        self._parallel = self._pool is not None
        if not self._parallel:
            self.workers = 1
        else:
            self._backend_used = self._pool.backend

    def _open_pool(self, workers: int) -> SectorPool | None:
        plan = self.plan
        if plan is None or not plan.is_parallel:
            return None
        pool = SectorPool(plan.backend, workers, logger=print)
        if not pool.active:
            return None
        return pool

    def _active_backend(self) -> str:
        return self._backend_used

    def _disable_pool(self, exc: BaseException) -> None:
        print(
            f"  [parallel] worker pool failed "
            f"({type(exc).__name__}: {exc}); continuing single-threaded"
        )
        if self._pool is not None:
            self._pool.close()
            self._pool = None
        self._parallel = False
        self.workers = 1
        self._backend_used = "serial"

    # ------------------------------------------------------------------ #
    # One round
    # ------------------------------------------------------------------ #
    def _run_round(
        self, sectors, round_index: int, round_seed: int
    ) -> list[SectorOutcome] | None:
        parallel = self._pool is not None and self._pool.active

        tasks = [
            SectorTask(
                sector=sector,
                seed=derive_seed(round_seed, sector.index),
                round_index=round_index,
                sectors_in_round=len(sectors),
                capture_frames=bool(self.visualizers),
                max_ticks=self.engine.max_ticks,
                db_config=self.db.config if parallel else None,
            )
            for sector in sectors
        ]

        outcomes: list[SectorOutcome] | None = None
        if parallel:
            try:
                outcomes = self._pool.map(tasks)
            except POOL_FAILURES as exc:
                self._disable_pool(exc)
                return None

        if outcomes is None:
            outcomes = [self._run_sector_serial(task) for task in tasks]

        # ``executor.map`` preserves submission order, but sorting makes the
        # determinism explicit: reports, frames and float sums are always in
        # sector order regardless of which worker finished first.
        outcomes.sort(key=lambda outcome: outcome.report.index)
        return outcomes

    def _run_sector_serial(self, task: SectorTask) -> SectorOutcome:
        sector = task.sector
        self._load_sector(sector)
        try:
            outcome = self.engine.run(
                sector,
                self.humans,
                self.zombies,
                random.Random(task.seed),
                round_index=task.round_index,
                sectors_in_round=task.sectors_in_round,
            )
        except BaseException:
            self.humans.clear()
            self.zombies.clear()
            raise
        self._unload_sector()
        return outcome

    def _record_round(self, outcomes, sector_count: int, round_index: int) -> None:
        for outcome in outcomes:
            report = outcome.report
            self.report.sectors.append(report)

            # Frames are replayed on the main thread, in sector order: neither
            # visualiser is safe to touch from a worker.
            for snapshot, meta in outcome.frames:
                for visualizer in self.visualizers:
                    visualizer.frame(snapshot, meta)

            if self.on_sector_complete is not None:
                self.on_sector_complete(report, sector_count, round_index)

    # ------------------------------------------------------------------ #
    # Loading / unloading (single-threaded path)
    # ------------------------------------------------------------------ #
    def _load_sector(self, sector) -> None:
        humans, zombies = self.db.load_agents(sector.agent_ids)
        self.humans = {agent.agent_id: agent for agent in humans}
        self.zombies = {agent.agent_id: agent for agent in zombies}

    def _unload_sector(self) -> None:
        """Write state back to PostgreSQL, then release the Python objects."""
        everything = [*self.humans.values(), *self.zombies.values()]
        if everything:
            self.db.save_agents(everything)
        self.humans.clear()
        self.zombies.clear()

    # ------------------------------------------------------------------ #
    # Visualiser plumbing
    # ------------------------------------------------------------------ #
    def _start_visualizers(self, sector_count: int, agent_count: int) -> None:
        info = {
            "backend": self._active_backend(),
            "workers": self.workers,
            "cpu_count": self.plan.cpu_count if self.plan else 1,
            "sectors": sector_count,
            "seed": self.seed,
        }
        for visualizer in self.visualizers:
            visualizer.start(sector_count, agent_count, info=info)
