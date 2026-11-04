"""Entry point for the zombie simulation.

Run with:  python main.py
"""

from __future__ import annotations

import argparse
import getpass
import random
import sys
import webbrowser
from pathlib import Path

import psycopg

from config.db_config import (
    CONFIG_PATH,
    describe_db_config,
    ensure_config_file,
    load_db_config,
    save_db_config,
)
from config import simulation_config as cfg
from db import schema
from db.db_manager import DBManager
from runtime.capabilities import VALID_BACKENDS, resolve_plan
from src.simulation import Simulation
from src.spawner import generate_spawns
from vis.html_vis import HtmlVisualizer

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Zombie simulation (Python + PostgreSQL).",
        epilog="Run without arguments for the interactive menu.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="start a simulation immediately instead of showing the menu",
    )
    parser.add_argument("--humans", type=int, default=40,
                        help="humans to spawn (--run)")
    parser.add_argument("--zombies", type=int, default=40,
                        help="zombies to spawn (--run)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed (--run)")
    parser.add_argument(
        "--mode",
        choices=("html", "headless"),
        default="html",
        help="html writes a player and opens it; headless prints the summary only",
    )
    parser.add_argument(
        "--headless", action="store_true", help="shorthand for --mode headless"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="worker processes/threads; 0 = auto (one per usable CPU)",
    )
    parser.add_argument(
        "--backend",
        choices=VALID_BACKENDS,
        default=None,
        help="parallel backend; auto picks processes on normal CPython",
    )
    parser.add_argument(
        "--sectors",
        type=int,
        default=None,
        help=(
            "sectors per round; 0 = auto (one per AGENTS_PER_SECTOR agents). "
            "Pin this to force a particular split"
        ),
    )
    parser.add_argument(
        "--event-shards",
        type=int,
        default=None,
        help=(
            "per-tick shards inside a sector; 0 = auto. Only used by the "
            "thread backend (free-threaded Python or --backend thread)"
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def print_banner() -> None:
    print("=" * 66)
    print("  ZOMBIE SIMULATION  -  Python + PostgreSQL")
    print(f"  project : {PROJECT_ROOT}")
    print(f"  db login: {CONFIG_PATH}")
    print("=" * 66)


def print_menu(config: dict) -> None:
    print()
    print(f"Database: {describe_db_config(config)}")
    print("  1) Start new simulation")
    print("  2) Clear & recreate database")
    print("  3) Database settings")
    print("  4) Database status")
    print("  5) Exit (clears agents)")
    print()


def ask_int(prompt: str, default: int, minimum: int = 0, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a whole number.")
            continue
        if value < minimum:
            print(f"  Must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"  Must be at most {maximum}.")
            continue
        return value


def ask_choice(prompt: str, default: int, minimum: int, maximum: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if minimum <= value <= maximum:
            return value
        print(f"  Choose between {minimum} and {maximum}.")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes"}


# --------------------------------------------------------------------------- #
# Database helpers
# --------------------------------------------------------------------------- #
def _print_db_error(exc: psycopg.Error) -> None:
    print()
    print("Could not reach PostgreSQL:")
    print(f"  {str(exc).strip()}")
    print("Check that the server is running and the settings are correct")
    print("(option 3), then try again.")


def ensure_database(config: dict) -> bool:
    """Create the configured database if needed; print a friendly error."""
    try:
        schema.ensure_database(config)
    except psycopg.Error as exc:
        _print_db_error(exc)
        return False
    return True


def try_bootstrap(config: dict) -> bool:
    try:
        missing = schema.bootstrap(config)
        if missing:
            print()
            print(missing)
            return False
        stale_bounds = schema.map_bounds_mismatch(config)
    except psycopg.Error as exc:
        _print_db_error(exc)
        return False
    if stale_bounds:
        print()
        print(stale_bounds)
        print("Choose 2) Clear & recreate database once, then try again.")
        return False
    return True


def handle_settings(config: dict) -> dict:
    print()
    print("Database settings (press Enter to keep the current value)")
    updated = dict(config)

    for key, label in (
        ("host", "Host"),
        ("port", "Port"),
        ("dbname", "Database name"),
        ("user", "User"),
    ):
        raw = input(f"  {label} [{updated[key]}]: ").strip()
        if not raw:
            continue
        if key == "port":
            try:
                updated[key] = int(raw)
            except ValueError:
                print("  Port must be a number - keeping the previous value.")
        else:
            updated[key] = raw

    password = getpass.getpass(
        "  Password [hidden, blank = keep, '-' = clear]: "
    )
    if password == "-":
        updated["password"] = ""
    elif password:
        updated["password"] = password

    if not try_bootstrap(updated):
        print("Not saved - the database test failed (see above).")
        return config
    saved_to = save_db_config(updated)
    print(f"Saved to {saved_to}")
    print("Connection OK.")
    return updated


def handle_reset(config: dict) -> None:
    print()
    if not ask_yes_no(
        "Drop and recreate agents/humans/zombies tables (all agents are lost)?",
        default=False,
    ):
        print("Cancelled.")
        return
    if not ensure_database(config):
        return
    db = DBManager(config)
    try:
        db.connect()
        if not db.try_run_lock():
            print("A simulation is currently using this database. Stop it first.")
            return
        schema.reset_database(config)
        print("Database cleared and recreated.")
    except psycopg.Error as exc:
        print(f"Reset failed: {exc}")
    finally:
        db.close()


def handle_status(config: dict) -> None:
    if not try_bootstrap(config):
        return
    try:
        with DBManager(config) as db:
            stats = db.stats()
    except psycopg.Error as exc:
        print(f"Could not read status: {exc}")
        return

    humans = stats["humans"]
    zombies = stats["zombies"]
    print()
    print(f"  humans : {humans['alive']} alive / {humans['dead']} dead")
    print(f"  zombies: {zombies['alive']} alive / {zombies['dead']} dead")


def handle_exit(config: dict) -> None:
    db = DBManager(config)
    try:
        db.connect()
        if not db.try_run_lock():
            print("A simulation is using the database; its agents were left alone.")
            return
        removed = schema.clear_agents(config)
        print(f"Cleared {removed} agent(s) from the database. Goodbye.")
    except psycopg.Error as exc:
        print(f"Could not clear agents: {exc}")
    finally:
        db.close()


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def handle_simulation(config: dict, args: argparse.Namespace | None = None) -> bool:
    # The database must exist before DBManager.connect(); the rest of the DDL
    # runs after the run lock is held so a second starter cannot race it.
    if not ensure_database(config):
        return False

    batch = bool(args is not None and args.run)
    print()

    if batch:
        humans = max(0, args.humans)
        zombies = max(0, args.zombies)
    else:
        humans = ask_int(
            "Number of humans", default=40, minimum=0, maximum=cfg.MAX_AGENTS
        )
        zombies = ask_int(
            "Number of zombies", default=40, minimum=0, maximum=cfg.MAX_AGENTS
        )
    total = humans + zombies
    if total == 0:
        print("Nothing to simulate - at least one agent is required.")
        return False
    if total > cfg.MAX_AGENTS:
        print(
            f"Too many agents: {total} requested, the limit is {cfg.MAX_AGENTS}."
        )
        return False

    if batch:
        mode = "headless" if args.headless else args.mode
    else:
        print()
        print("Visualisation:")
        print("  1) HTML file (opens in your browser)")
        print("  2) Headless (summary only)")
        choice = ask_choice("Choice", default=1, minimum=1, maximum=2)
        mode = "html" if choice == 1 else "headless"

    if batch:
        seed = args.seed
    else:
        seed_raw = input("Random seed (blank = random): ").strip()
        if seed_raw:
            try:
                seed = int(seed_raw)
            except ValueError:
                print("  Seed must be a whole number - using a random seed instead.")
                seed = random.randrange(1_000_000)
                print(f"  Using seed {seed}.")
        else:
            seed = None
    rng = random.Random(seed) if seed is not None else random.Random()

    workers = args.workers if args is not None else None
    backend = args.backend if args is not None else None
    sectors = args.sectors if args is not None else None
    event_shards = args.event_shards if args is not None else None
    plan = resolve_plan(workers, backend)
    print(f"  Parallel: {plan.describe()}")
    if sectors:
        print(f"  Sectors per round: {sectors} (pinned)")
    else:
        print(
            "  Sectors per round: auto "
            f"(one per {cfg.AGENTS_PER_SECTOR} agents, "
            f"max {cfg.MAX_SECTORS})"
        )
    if plan.backend == "thread":
        print(
            "  Event shards: "
            + (f"{event_shards} (pinned)" if event_shards else "auto")
        )

    visualizers = []
    html_visualizer = None
    if mode == "html":
        html_visualizer = HtmlVisualizer()
        visualizers.append(html_visualizer)

    db = DBManager(config)
    simulation: Simulation | None = None
    try:
        db.connect()
        if not db.try_run_lock():
            print()
            print("Another simulation is already using this database.")
            print("Stop it before starting a new run.")
            return False
        if not try_bootstrap(config):
            return False
        removed = schema.clear_agents(config)
        if removed:
            print(
                f"Removed {removed} leftover agent(s) from the previous run.")

        specs = generate_spawns(humans, zombies, rng)

        def progress(sector_report, total: int, round_index: int) -> None:
            print(
                f"  round {round_index} sector {sector_report.index}/{total}: "
                f"humans {sector_report.humans_start}->{sector_report.humans_end}, "
                f"zombies {sector_report.zombies_start}->{sector_report.zombies_end}, "
                f"{sector_report.iterations} tick(s), "
                f"{sector_report.stop_reason}"
            )

        simulation = Simulation(
            db,
            rng=rng,
            visualizers=visualizers,
            on_sector_complete=progress,
            workers=workers,
            backend=backend,
            sectors=sectors,
            event_shards=event_shards,
            seed=seed,
        )
        # Let worker start-up overlap with the insert instead of delaying the
        # first round (a no-op when the run stays single-threaded).
        simulation.prewarm(len(specs))
        db.insert_agents(specs)
        print(f"Spawned {humans} human(s) and {zombies} zombie(s).")

        report = simulation.run()
        print(report.render())

        if html_visualizer is not None:
            print(f"HTML visualisation: {html_visualizer.path}")
            try:
                webbrowser.open(html_visualizer.path.resolve().as_uri())
            except Exception:
                print("Open the file manually in a browser to watch the run.")
        return True
    finally:
        if simulation is not None:
            simulation.close()
        db.close()


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ensure_config_file()
    print_banner()
    config = load_db_config()

    if args.run:
        try:
            ok = handle_simulation(config, args)
        except KeyboardInterrupt:
            print()
            print("Interrupted - clearing agents from the database.")
            handle_exit(config)
            return 130
        except RuntimeError as exc:
            print()
            print(f"Run aborted: {exc}")
            handle_exit(config)
            return 1
        return 0 if ok else 1

    try_bootstrap(config)

    while True:
        print_menu(config)
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            handle_exit(config)
            return 0

        try:
            if choice == "1":
                handle_simulation(config, args)
            elif choice == "2":
                handle_reset(config)
            elif choice == "3":
                config = handle_settings(config)
            elif choice == "4":
                handle_status(config)
            elif choice in {"5", "q", "quit", "exit"}:
                handle_exit(config)
                return 0
            else:
                print("Unknown option.")
        except KeyboardInterrupt:
            print()
            print("Interrupted - clearing agents from the database.")
            handle_exit(config)
            return 130
        except RuntimeError as exc:
            print()
            print(f"Run aborted: {exc}")
            print("The next run clears the leftover agents automatically.")


if __name__ == "__main__":
    sys.exit(main())
