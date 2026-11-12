"""Measure what parallel sectors are actually worth on this machine.

Runs the real sector engine on synthetic agents - no database, no
visualisers - so the numbers reflect CPU scaling only.  Use it to pick a
``--workers`` value for a specific chip: hybrid big.LITTLE parts, SMT-heavy
desktops and small ARM boards all peak at different counts.

The sector count comes from the population, so it caps the outer pool:
1200 agents make 3 sectors and use at most 3 outer workers. On the thread
backend the rest of the ``--workers`` budget becomes per-sector shards, and the
benchmark resolves both from the same helpers as a real run.

``--accel`` picks the nearest-enemy backend for the whole measurement, so the
same population can be compared across ``python`` and ``numpy``.

    python tools/bench_parallel.py
    python tools/bench_parallel.py --population 4000 --workers 1,2,4,8,12
    python tools/bench_parallel.py --accel numpy
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
from runtime import accelerator as accel_mod  # noqa: E402
from runtime.capabilities import (  # noqa: E402
    resolve_event_shards,
    resolve_plan,
    resolve_workers,
)
from runtime.sector_pool import SectorPool  # noqa: E402
from src.sector_task import build_sector_tasks, simulate_sector_task  # noqa: E402
from src.spawner import build_agents  # noqa: E402
from utils.sectoring import build_sectors, resolve_sector_count  # noqa: E402


def _with_shards(tasks, event_shards: int):
    cloned = copy.deepcopy(tasks)
    for task in cloned:
        task.event_shards = event_shards
    return cloned


def measure_serial(pristine, rounds: int) -> tuple[float, int]:
    """Every timed round starts from the same state, for both backends."""
    started = time.perf_counter()
    ticks = 0
    for _ in range(max(1, rounds)):
        tasks = _with_shards(pristine, 1)
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
        try:
            values.append(int(piece))
        except ValueError:
            raise ValueError(
                f"--workers expects comma-separated whole numbers; got {piece!r}"
            ) from None
    return values or [1]


def _row_plan(
    requested_workers: int, sector_count: int, backend: str, event_shards: int
) -> tuple[int, int, int]:
    """Resolve a row the same way a real run does.

    Returns ``(worker budget, outer pool size, per-sector shards)``.
    """
    budget = resolve_workers(requested_workers)
    outer = min(budget, max(1, sector_count))
    shards = resolve_event_shards(sector_count, budget, backend, event_shards)
    return budget, outer, shards


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
        help="comma-separated worker budgets to test (default 1,2,4,8)",
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
        help="per-sector tick shards for the pooled rows (0 = auto)",
    )
    parser.add_argument(
        "--accel",
        choices=("auto", "off", "numpy"),
        default=None,
        help=(
            "nearest-enemy backend for every row (default: the configured "
            "auto choice)"
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    try:
        worker_budgets = parse_worker_list(args.workers)
    except ValueError as exc:
        print(exc)
        return 1

    if args.population <= 0:
        print("Population must be a positive number of agents.")
        return 1

    if not 0.0 <= args.split <= 1.0:
        print(f"Split must be a fraction between 0 and 1; got {args.split}.")
        return 1

    if args.population > cfg.MAX_AGENTS:
        print(
            f"Population {args.population} exceeds MAX_AGENTS="
            f"{cfg.MAX_AGENTS}; refusing to allocate it."
        )
        return 1

    humans = max(0, int(round(args.population * args.split)))
    zombies = max(0, args.population - humans)
    agents = build_agents(humans, zombies, random.Random(args.seed))

    plan = resolve_plan()
    accel_plan = accel_mod.configure(args.accel)
    accel_mode = accel_mod.mode()
    print()
    print(f"Plan      : {plan.describe()}")
    print(f"Accel     : {accel_plan.describe()}")
    print(f"Population: {humans} humans / {zombies} zombies ({len(agents)} total)")
    print(f"Tick cap  : runs are bounded by MAX_TOTAL_TICKS={cfg.MAX_TOTAL_TICKS}")

    # Clustering itself is serial between rounds; report it once so a slow
    # sectoring step cannot be mistaken for poor worker scaling.
    positions = {a.agent_id: (float(a.x_pos), float(a.y_pos)) for a in agents}
    types = {a.agent_id: a.agent_type for a in agents}
    count = resolve_sector_count(len(agents), requested=args.sectors)
    started = time.perf_counter()
    build_sectors(positions, target_k=count, types=types)
    cluster_ms = (time.perf_counter() - started) * 1000

    pristine = build_sector_tasks(agents, args.seed, count, accel=accel_mode)
    sector_count = len(pristine)
    sizes = [len(t.humans or []) + len(t.zombies or []) for t in pristine]
    print(
        f"Sectors   : {sector_count} per round, "
        f"{min(sizes)}..{max(sizes)} agents each"
    )
    print(f"Clustering: {cluster_ms:.1f} ms per round (serial)")
    print(f"Shards    : {args.event_shards or 'auto'} per sector (budget-capped)")
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
    measured_plans: set[tuple[int, int]] = set()
    for requested_workers in worker_budgets:
        if requested_workers <= 1:
            continue
        if args.backend and args.backend != "auto":
            backend = args.backend
        else:
            backend = "thread" if not plan.gil_enabled else "process"
        budget, workers, event_shards = _row_plan(
            requested_workers, sector_count, backend, args.event_shards
        )
        if (workers, event_shards) in measured_plans:
            continue
        measured_plans.add((workers, event_shards))
        if workers <= 1:
            print(
                f"{'serial':<9} {budget:>7} {rounds:>6} {'-':>7} "
                f"{'-':>8} {'-':>9} {'-':>8} {'-':>10} {'-':>8}"
                "  (only one sector)"
            )
            continue
        best = None
        for _ in range(max(1, args.repeats)):
            measured = measure_pool(
                pristine,
                backend,
                workers,
                rounds,
                event_shards=event_shards,
            )
            if measured is None:
                best = None
                break
            if best is None or measured[0] < best[0]:
                best = measured
        if best is None:
            print(f"{backend:<9} {budget:>7} {'pool could not start':>46}")
            continue

        elapsed, ticks, startup = best
        speedup = baseline[0] / elapsed if elapsed > 0 else 0.0
        efficiency = speedup / budget
        results.append((backend, budget, ticks, elapsed, speedup, efficiency))
        if budget != requested_workers:
            note = f"  (capped at MAX_WORKERS={cfg.MAX_WORKERS})"
        elif workers < budget:
            note = "  (outer capped by sectors)"
        else:
            note = ""
        print(
            f"{backend:<9} {budget:>7} {rounds:>6} {ticks:>7} "
            f"{elapsed:>8.2f} {ticks / elapsed:>9.1f} {speedup:>7.2f}x "
            f"{efficiency:>9.0%} {startup:>7.2f}s{note}"
        )

    if not results:
        print(
            "No pooled row was measured (single sector or pool unavailable); "
            "this machine will run single-threaded."
        )
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
