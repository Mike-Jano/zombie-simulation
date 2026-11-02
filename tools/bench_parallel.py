"""Measure what parallel sectors are actually worth on this machine.

Runs the real sector engine on synthetic agents - no database, no
visualisers - so the numbers reflect CPU scaling only.  Use it to pick a
``--workers`` value for a specific chip: hybrid big.LITTLE parts, SMT-heavy
desktops and small ARM boards all peak at different counts.

The sector count comes from the population, so it also caps the useful worker
count: 1200 agents make 3 sectors and cannot use more than 3 workers.  Rows
that would exceed it are capped and labelled.

    python tools/bench_parallel.py
    python tools/bench_parallel.py --population 4000 --workers 1,2,4,8,12
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from runtime.capabilities import resolve_plan  # noqa: E402
from runtime.sector_pool import SectorPool  # noqa: E402
from src.sector_engine import derive_seed  # noqa: E402
from src.sector_task import SectorTask, simulate_sector_task  # noqa: E402
from src.spawner import build_agents  # noqa: E402
from utils.sectoring import build_sectors, resolve_sector_count  # noqa: E402


def build_tasks(
    agents, seed: int, sectors: int | None = None, event_shards: int = 1
):
    positions = {
        agent.agent_id: (float(agent.x_pos), float(agent.y_pos)) for agent in agents
    }
    types = {agent.agent_id: agent.agent_type for agent in agents}
    count = resolve_sector_count(len(agents), requested=sectors)
    sector_list = build_sectors(
        positions, rng=random.Random(seed), target_k=count, types=types
    )

    by_id = {agent.agent_id: agent for agent in agents}
    tasks = []
    for sector in sector_list:
        members = [by_id[agent_id] for agent_id in sector.agent_ids]
        tasks.append(
            SectorTask(
                sector=sector,
                seed=derive_seed(seed, sector.index),
                round_index=1,
                sectors_in_round=len(sector_list),
                capture_frames=False,
                event_shards=event_shards,
                humans=[a for a in members if a.agent_type == "human"],
                zombies=[a for a in members if a.agent_type == "zombie"],
            )
        )
    return tasks, len(sector_list)


def _with_shards(tasks, event_shards: int):
    cloned = copy.deepcopy(tasks)
    for task in cloned:
        task.event_shards = event_shards
    return cloned


def measure_serial(pristine, rounds: int, event_shards: int = 1) -> tuple[float, int]:
    """Every timed round starts from the same state, for both backends."""
    started = time.perf_counter()
    ticks = 0
    for _ in range(max(1, rounds)):
        tasks = _with_shards(pristine, event_shards)
        outcomes = [simulate_sector_task(task) for task in tasks]
        ticks += sum(o.report.iterations for o in outcomes)
    return time.perf_counter() - started, ticks


def measure_pool(
    pristine, backend: str, workers: int, rounds: int, event_shards: int = 1
):
    """Steady-state throughput: one pool, many rounds (like a real run)."""
    pool = SectorPool(backend, workers)
    if not pool.active:
        return None
    try:
        # Warm-up costs are reported separately: pool start + first dispatch.
        warm_started = time.perf_counter()
        pool.map(_with_shards(pristine, event_shards))
        startup = time.perf_counter() - warm_started

        started = time.perf_counter()
        ticks = 0
        for _ in range(max(1, rounds)):
            outcomes = pool.map(_with_shards(pristine, event_shards))
            ticks += sum(o.report.iterations for o in outcomes)
        elapsed = time.perf_counter() - started
    finally:
        pool.close()
    return elapsed, ticks, startup


def parse_worker_list(raw: str) -> list[int]:
    values: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        values.append(int(piece))
    return values or [1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--population", type=int, default=1200, help="total agents (default 1200)"
    )
    parser.add_argument(
        "--split", type=float, default=0.5, help="fraction of humans (default 0.5)"
    )
    parser.add_argument(
        "--workers",
        default="1,2,4,8",
        help="comma-separated worker counts to test (default 1,2,4,8)",
    )
    parser.add_argument(
        "--sectors", type=int, default=None, help="pin the sector count (0 = auto)"
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="timed repetitions (default 2)"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=8,
        help="rounds per repetition, one pool (default 8)",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "thread", "process"),
        default=None,
        help="force a pool backend (default: auto from the GIL probe)",
    )
    parser.add_argument(
        "--event-shards",
        type=int,
        default=0,
        help="per-sector tick shards for the pooled rows (0 = one)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    humans = max(0, int(round(args.population * args.split)))
    zombies = max(0, args.population - humans)
    agents = build_agents(humans, zombies, random.Random(args.seed))

    plan = resolve_plan()
    print()
    print(f"Plan      : {plan.describe()}")
    print(f"Population: {humans} humans / {zombies} zombies ({len(agents)} total)")
    print(f"Tick cap  : runs are bounded by MAX_TOTAL_TICKS={cfg.MAX_TOTAL_TICKS}")

    # Clustering itself is serial between rounds; report it once so a slow
    # sectoring step cannot be mistaken for poor worker scaling.
    positions = {a.agent_id: (float(a.x_pos), float(a.y_pos)) for a in agents}
    types = {a.agent_id: a.agent_type for a in agents}
    count = resolve_sector_count(len(agents), requested=args.sectors)
    started = time.perf_counter()
    build_sectors(
        positions, rng=random.Random(args.seed), target_k=count, types=types
    )
    cluster_ms = (time.perf_counter() - started) * 1000

    pristine, sector_count = build_tasks(
        agents,
        args.seed,
        sectors=args.sectors,
        event_shards=max(1, args.event_shards),
    )
    sizes = [len(t.humans or []) + len(t.zombies or []) for t in pristine]
    print(
        f"Sectors   : {sector_count} per round, "
        f"{min(sizes)}..{max(sizes)} agents each"
    )
    print(f"Clustering: {cluster_ms:.1f} ms per round (serial)")
    print(f"Shards    : {max(1, args.event_shards)} per sector in pooled rows")
    print()

    header = (
        f"{'backend':<9} {'workers':>7} {'rounds':>6} {'ticks':>7} "
        f"{'seconds':>8} {'ticks/s':>9} {'speedup':>8} {'efficiency':>10} "
        f"{'startup':>8}"
    )
    print(header)
    print("-" * len(header))

    rounds = max(1, args.rounds)

    # Serial baseline: best of N, so a single scheduling hiccup cannot inflate
    # every speedup number.
    baseline = None
    for _ in range(max(1, args.repeats)):
        elapsed, ticks = measure_serial(pristine, rounds)
        if baseline is None or elapsed < baseline[0]:
            baseline = (elapsed, ticks)

    elapsed, ticks = baseline
    print(
        f"{'serial':<9} {1:>7} {rounds:>6} {ticks:>7} "
        f"{elapsed:>8.2f} {ticks / elapsed:>9.1f} {1.0:>7.2f}x "
        f"{1.0:>9.0%} {'-':>8}"
    )

    results = []
    measured_widths: set[int] = set()
    for requested_workers in parse_worker_list(args.workers):
        if requested_workers <= 1:
            continue
        # A real run never uses more workers than sectors, so measuring more is
        # meaningless: the extra processes would sit idle.
        workers = min(requested_workers, sector_count)
        if workers in measured_widths:
            continue
        measured_widths.add(workers)
        if workers <= 1:
            print(
                f"{'serial':<9} {1:>7} {rounds:>6} {'-':>7} "
                f"{'-':>8} {'-':>9} {'-':>8} {'-':>10} {'-':>8}"
                "  (only one sector)"
            )
            continue
        if args.backend and args.backend != "auto":
            backend = args.backend
        else:
            backend = "thread" if not plan.gil_enabled else "process"
        best = None
        for _ in range(max(1, args.repeats)):
            measured = measure_pool(
                pristine,
                backend,
                workers,
                rounds,
                event_shards=max(1, args.event_shards),
            )
            if measured is None:
                best = None
                break
            if best is None or measured[0] < best[0]:
                best = measured
        if best is None:
            print(f"{backend:<9} {workers:>7} {'pool could not start':>46}")
            continue

        elapsed, ticks, startup = best
        speedup = baseline[0] / elapsed if elapsed > 0 else 0.0
        efficiency = speedup / workers
        results.append((backend, workers, ticks, elapsed, speedup, efficiency))
        note = "" if requested_workers == workers else "  (capped by sectors)"
        print(
            f"{backend:<9} {workers:>7} {rounds:>6} {ticks:>7} "
            f"{elapsed:>8.2f} {ticks / elapsed:>9.1f} {speedup:>7.2f}x "
            f"{efficiency:>9.0%} {startup:>7.2f}s{note}"
        )

    if not results:
        print("No pool could be started; this machine will run single-threaded.")
        return 0

    serial_seconds = baseline[0]
    serial_ticks = baseline[1]
    print()
    print(
        f"Serial baseline: {serial_ticks} ticks in {serial_seconds:.2f}s "
        f"({serial_ticks / serial_seconds:.1f} ticks/s)"
    )

    best = max(results, key=lambda row: row[2] / row[3])
    print(
        f"Suggested     : --workers {best[1]} "
        f"({best[0]} backend, {best[2] / best[3]:.1f} ticks/s, "
        f"{best[4]:.2f}x)"
    )
    projected = cfg.MAX_TOTAL_TICKS / (best[2] / best[3])
    print(
        f"Projected full run ({cfg.MAX_TOTAL_TICKS} world ticks): "
        f"~{projected:.1f}s of sector work"
    )
    print(
        "Note: the visualisation is optional; measure with --mode headless "
        "(fastest) or --mode html."
    )
    if sector_count <= 1:
        print(
            f"Note: {len(agents)} agents produce a single sector, so a real run "
            "is single-threaded. Raise the population (or lower "
            "AGENTS_PER_SECTOR) to use workers."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
