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
from runtime import accelerator as accel_mod
from runtime.capabilities import VALID_BACKENDS, resolve_plan
from src.simulation import Simulation
from src.spawner import generate_spawns, promote_champions
from utils.sectoring import map_bounds_for, max_sectors_for
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
        "--carry-champions",
        action="store_true",
        help=(
            "carry the previous run's survivors over (--run); counts are "
            "raised to at least the champion total"
        ),
    )
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
            "sectors per round; 0 = auto (one per AGENTS_PER_SECTOR agents, "
            "capped by the starting population). Pin this to force a split"
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
    parser.add_argument(
        "--accel",
        choices=("auto", "off", "numpy"),
        default=None,
        help=(
            "nearest-enemy acceleration; auto = numpy when installed (default)"
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
    print("  5) Exit (keeps champions)")
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


def check_database(config: dict) -> tuple[bool, str | None]:
    """Test a login: ``(connection works, missing-schema message)``.

    A stale schema is not a bad login: the reset (option 2) needs these
    credentials, so callers that save settings must distinguish the two.
    """
    try:
        return True, schema.bootstrap(config)
    except psycopg.Error as exc:
        _print_db_error(exc)
        return False, None


def try_bootstrap(config: dict) -> bool:
    ok, missing = check_database(config)
    if missing:
        print()
        print(missing)
    return ok and not missing


def load_champions(config: dict) -> list[dict]:
    """Read the previous run's survivors.

    Only an old schema (no champion columns) counts as "no champions"; any
    other database error must reach the caller, because a run that cannot read
    the champions would otherwise clear them.
    """
    db = DBManager(config)
    try:
        db.connect()
        return db.load_champions()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        return []
    finally:
        db.close()


def load_champion_counts(config: dict) -> dict[str, int]:
    """Count the previous run's survivors by side, without loading the rows."""
    db = DBManager(config)
    try:
        db.connect()
        return db.count_champions()
    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
        return {}
    finally:
        db.close()


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

    ok, missing = check_database(updated)
    if not ok:
        print("Not saved - the database test failed (see above).")
        return config
    saved_to = save_db_config(updated)
    print(f"Saved to {saved_to}")
    print("Connection OK.")
    if missing:
        print()
        print(missing)
        print("The settings were saved, so option 2 can recreate the schema now.")
    return updated


def handle_reset(config: dict) -> None:
    print()
    if not ask_yes_no(
        "Drop and recreate agents/humans/zombies tables "
        "(all agents and champions are lost)?",
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
        removed = schema.clear_agents(config, keep_champions=True)
        print(
            f"Cleared {removed} agent(s) from the database "
            "(champions kept for the next run). Goodbye."
        )
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

    # The carry-over prompt must come before the counts, because the champion
    # totals become the defaults and the lower bound for each side.
    carry = bool(getattr(args, "carry_champions", False)) if batch else False
    counts: dict[str, int] = {}
    if not batch:
        counts = load_champion_counts(config)
        if counts:
            print(
                f"Champions from the previous run: "
                f"{counts.get('human', 0)} human(s), "
                f"{counts.get('zombie', 0)} zombie(s)."
            )
            carry = ask_yes_no("Carry the champions into this run?", default=False)

    # Full rows are only needed once the user has opted in.
    champions: list[dict] = load_champions(config) if carry else []
    human_champions = sum(
        1 for champion in champions if champion["agent_type"] == "human"
    )
    zombie_champions = len(champions) - human_champions
    if human_champions > cfg.MAX_AGENTS or zombie_champions > cfg.MAX_AGENTS:
        print(
            f"Too many champions: the limit is {cfg.MAX_AGENTS} per side."
        )
        return False

    if batch:
        humans = max(max(0, args.humans), human_champions)
        zombies = max(max(0, args.zombies), zombie_champions)
        if humans != max(0, args.humans) or zombies != max(0, args.zombies):
            print(
                f"  Raised to the champion minimum: "
                f"{humans} human(s), {zombies} zombie(s)."
            )
    else:
        humans = ask_int(
            "Number of humans",
            default=human_champions or 40,
            minimum=human_champions,
            maximum=cfg.MAX_AGENTS,
        )
        zombies = ask_int(
            "Number of zombies",
            default=zombie_champions or 40,
            minimum=zombie_champions,
            maximum=cfg.MAX_AGENTS,
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
    accel = getattr(args, "accel", None)
    map_bounds = map_bounds_for(total)
    run_max_sectors = max_sectors_for(total)
    plan = resolve_plan(workers, backend)
    accel_plan = accel_mod.probe(accel)
    print(f"  Parallel: {plan.describe()}")
    print(f"  Accel: {accel_plan.describe()}")
    print(f"  Map: {int(map_bounds[0])}..{int(map_bounds[2])} on each axis")
    if sectors:
        print(
            f"  Sectors per round: {min(sectors, run_max_sectors)} "
            f"(pinned; max {run_max_sectors})"
        )
    else:
        print(
            "  Sectors per round: auto "
            f"(one per {cfg.AGENTS_PER_SECTOR} agents, "
            f"max {run_max_sectors} from {total} starting agents)"
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

    db = DBManager(config, map_bounds=map_bounds)
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
        # The prompt ran before the lock, so another process may have changed
        # the champions since. Carrying promotes the locked rows or nothing;
        # declining must not clear a set the user never saw.
        stale = False
        fresh: list[dict] = []
        if carry:
            fresh = db.load_champions()
            stale = [champion["agent_id"] for champion in fresh] != [
                champion["agent_id"] for champion in champions
            ]
        elif counts:
            stale = db.count_champions() != counts
        if stale:
            print()
            print(
                "The champions changed while this run was being set up "
                "(another run finished or cleared them)."
            )
            print("Start the run again to use the current champions.")
            return False
        if carry:
            champions = fresh
        removed = schema.clear_agents(config, keep_champions=carry)
        if removed:
            print(
                f"Removed {removed} leftover agent(s) from the previous run.")

        # Champions are promoted in place; only the shortfall is newly spawned.
        promoted = (
            promote_champions(champions, rng, map_bounds=map_bounds)
            if carry
            else []
        )
        if promoted:
            db.save_champion_upgrades(promoted)
        new_humans = humans - human_champions
        new_zombies = zombies - zombie_champions
        specs = generate_spawns(
            new_humans, new_zombies, rng, map_bounds=map_bounds
        )

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
            accel=accel,
            seed=seed,
            map_bounds=map_bounds,
            max_sectors=run_max_sectors,
        )
        # Let worker start-up overlap with the insert instead of delaying the
        # first round (a no-op when the run stays single-threaded).
        simulation.prewarm(total)
        db.insert_agents(specs)
        print(
            f"Spawned {new_humans} human(s) and {new_zombies} zombie(s)."
        )
        if carry:
            print(
                f"Carried over {human_champions} human and "
                f"{zombie_champions} zombie champion(s)."
            )

        report = simulation.run()
        print(report.render())
        # Survivors become the next run's champions; dead rows lose the flag.
        db.mark_champions()

        if html_visualizer is not None:
            if html_visualizer.frames:
                print(f"HTML visualisation: {html_visualizer.path}")
                try:
                    webbrowser.open(html_visualizer.path.resolve().as_uri())
                except Exception:
                    print("Open the file manually in a browser to watch the run.")
            else:
                print("No HTML player: the run produced no frames.")
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
        except psycopg.Error as exc:
            print()
            _print_db_error(exc)
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
        except EOFError:
            print()
            handle_exit(config)
            return 0
        except KeyboardInterrupt:
            print()
            print("Interrupted - clearing agents from the database.")
            handle_exit(config)
            return 130
        except RuntimeError as exc:
            print()
            print(f"Run aborted: {exc}")
            print("The next run clears the leftover agents automatically.")
        except psycopg.Error as exc:
            print()
            _print_db_error(exc)
            print("The next run clears the leftover agents automatically.")


if __name__ == "__main__":
    sys.exit(main())
