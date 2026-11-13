"""CLI guard rails and the reset ordering that needs the database to exist."""

from __future__ import annotations

import argparse
import io
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


class AgentLimitTests(unittest.TestCase):
    def test_too_many_agents_is_rejected_before_connecting(self):
        args = argparse.Namespace(
            run=True, humans=main.cfg.MAX_AGENTS, zombies=1
        )
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ensure_database", return_value=True))
            db_class = stack.enter_context(patch.object(main, "DBManager"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertFalse(main.handle_simulation({}, args))

        db_class.assert_not_called()


class RunStartOrderTests(unittest.TestCase):
    def test_database_is_ensured_before_connecting(self):
        calls: list[str] = []

        class FakeDB:
            def __init__(self, _config, map_bounds=None):
                calls.append("db")

            def connect(self):
                calls.append("connect")
                raise RuntimeError("stop here")

            def close(self):
                pass

        def ensure(_config):
            calls.append("ensure")
            return True

        args = argparse.Namespace(
            run=True,
            humans=1,
            zombies=1,
            seed=1,
            mode="headless",
            headless=True,
            workers=1,
            backend="serial",
            sectors=None,
            event_shards=None,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main, "ensure_database", side_effect=ensure)
            )
            stack.enter_context(patch.object(main, "DBManager", FakeDB))
            stack.enter_context(redirect_stdout(io.StringIO()))
            with self.assertRaises(RuntimeError):
                main.handle_simulation({}, args)

        self.assertEqual(calls[:2], ["ensure", "db"])


class ExitCodeTests(unittest.TestCase):
    def _run(self, ok: bool) -> int:
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ensure_config_file"))
            stack.enter_context(patch.object(main, "print_banner"))
            stack.enter_context(patch.object(main, "load_db_config", return_value={}))
            stack.enter_context(
                patch.object(main, "handle_simulation", return_value=ok)
            )
            return main.main(["--run"])

    def test_run_returns_zero_on_success(self):
        self.assertEqual(self._run(True), 0)

    def test_run_returns_one_on_failure(self):
        self.assertEqual(self._run(False), 1)

    def test_run_aborted_returns_one(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ensure_config_file"))
            stack.enter_context(patch.object(main, "print_banner"))
            stack.enter_context(patch.object(main, "load_db_config", return_value={}))
            stack.enter_context(
                patch.object(
                    main, "handle_simulation", side_effect=RuntimeError("boom")
                )
            )
            exit_mock = stack.enter_context(patch.object(main, "handle_exit"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(main.main(["--run"]), 1)

        exit_mock.assert_called_once()

    def test_database_error_returns_one(self):
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ensure_config_file"))
            stack.enter_context(patch.object(main, "print_banner"))
            stack.enter_context(patch.object(main, "load_db_config", return_value={}))
            stack.enter_context(
                patch.object(
                    main,
                    "handle_simulation",
                    side_effect=main.psycopg.Error("server went away"),
                )
            )
            exit_mock = stack.enter_context(patch.object(main, "handle_exit"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(main.main(["--run"]), 1)

        exit_mock.assert_not_called()

    def test_eof_inside_a_prompt_exits_cleanly(self):
        config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "zombie_sim",
            "user": "postgres",
            "password": "",
        }
        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ensure_config_file"))
            stack.enter_context(patch.object(main, "print_banner"))
            stack.enter_context(
                patch.object(main, "load_db_config", return_value=config)
            )
            stack.enter_context(patch.object(main, "try_bootstrap"))
            stack.enter_context(patch("builtins.input", return_value="1"))
            stack.enter_context(
                patch.object(main, "handle_simulation", side_effect=EOFError)
            )
            exit_mock = stack.enter_context(patch.object(main, "handle_exit"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(main.main([]), 0)

        exit_mock.assert_called_once()


class ResetOrderTests(unittest.TestCase):
    def test_database_is_ensured_before_connecting(self):
        calls: list[str] = []

        class FakeDB:
            def __init__(self, _config):
                calls.append("db")

            def connect(self):
                calls.append("connect")

            def try_run_lock(self):
                calls.append("lock")
                return True

            def close(self):
                calls.append("close")

        def ensure(_config):
            calls.append("ensure")
            return True

        with ExitStack() as stack:
            stack.enter_context(patch.object(main, "ask_yes_no", return_value=True))
            stack.enter_context(
                patch.object(main, "ensure_database", side_effect=ensure)
            )
            stack.enter_context(patch.object(main, "DBManager", FakeDB))
            stack.enter_context(patch.object(main.schema, "reset_database"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            main.handle_reset({})

        self.assertEqual(calls[:3], ["ensure", "db", "connect"])


class SettingsOrderTests(unittest.TestCase):
    def _run(self, test_ok: bool, password: str = ""):
        config = {
            "host": "old",
            "port": 5432,
            "dbname": "d",
            "user": "u",
            "password": "p",
        }
        with patch("builtins.input", side_effect=["new", "5432", "d", "u"]), patch.object(
            main.getpass, "getpass", return_value=password
        ), patch.object(main, "try_bootstrap", return_value=test_ok), patch.object(
            main, "save_db_config", return_value=Path("x.ini")
        ) as save, redirect_stdout(io.StringIO()):
            result = main.handle_settings(config)
        return result, save

    def test_failed_test_does_not_save(self):
        result, save = self._run(False)
        save.assert_not_called()
        self.assertEqual(result["host"], "old")

    def test_successful_test_saves(self):
        result, save = self._run(True)
        save.assert_called_once()
        self.assertEqual(result["host"], "new")
        self.assertEqual(result["password"], "p")

    def test_dash_clears_the_password(self):
        result, save = self._run(True, password="-")
        save.assert_called_once()
        self.assertEqual(result["password"], "")


class ChampionFlowTests(unittest.TestCase):
    def test_carry_over_changes_defaults_minimums_and_spawn_shortfall(self):
        calls: dict = {"ask_int": []}

        class FakeDB:
            def __init__(self, _config, map_bounds=None):
                pass

            def connect(self):
                pass

            def try_run_lock(self):
                return True

            def load_champions(self):
                return champions

            def save_champion_upgrades(self, rows):
                calls["promoted"] = rows

            def insert_agents(self, specs):
                calls["spawned"] = specs

            def mark_champions(self):
                calls["marked"] = True

            def close(self):
                pass

        class FakeReport:
            def render(self):
                return "report"

        class FakeSimulation:
            def __init__(self, _db, **_kwargs):
                pass

            def prewarm(self, count):
                calls["prewarm"] = count

            def run(self):
                return FakeReport()

            def close(self):
                pass

        champions = [
            {
                "agent_id": i,
                "agent_type": "human",
                "health": 100,
                "speed": 3,
                "is_super": False,
                "weapon_tier": 3,
                "strength": None,
            }
            for i in (1, 2)
        ]

        def fake_ask_int(prompt, default, minimum=0, maximum=None):
            calls["ask_int"].append((default, minimum))
            return {0: 5, 1: 3}[len(calls["ask_int"]) - 1]

        def fake_spawns(humans, zombies, _rng, map_bounds=None):
            calls["new"] = (humans, zombies)
            return []

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main, "ensure_database", return_value=True)
            )
            stack.enter_context(
                patch.object(main, "load_champion_counts", return_value={"human": 2})
            )
            stack.enter_context(
                patch.object(main, "load_champions", return_value=champions)
            )
            stack.enter_context(patch.object(main, "ask_yes_no", return_value=True))
            stack.enter_context(
                patch.object(main, "ask_int", side_effect=fake_ask_int)
            )
            stack.enter_context(
                patch.object(main, "ask_choice", return_value=2)
            )
            stack.enter_context(patch("builtins.input", return_value=""))
            stack.enter_context(patch.object(main, "DBManager", FakeDB))
            stack.enter_context(
                patch.object(main, "try_bootstrap", return_value=True)
            )
            stack.enter_context(
                patch.object(main.schema, "clear_agents", return_value=0)
            )
            stack.enter_context(
                patch.object(main, "generate_spawns", side_effect=fake_spawns)
            )
            stack.enter_context(
                patch.object(main, "Simulation", FakeSimulation)
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            ok = main.handle_simulation({}, None)

        self.assertTrue(ok)
        # Human champions raise the human default/minimum; the zombie side
        # keeps the normal 40/0.
        self.assertEqual(calls["ask_int"][0], (2, 2))
        self.assertEqual(calls["ask_int"][1], (40, 0))
        self.assertEqual(calls["new"], (3, 3))
        self.assertEqual(calls["prewarm"], 8)
        self.assertEqual(len(calls["promoted"]), 2)
        self.assertTrue(calls["marked"])


class BatchChampionTests(unittest.TestCase):
    def test_counts_below_the_champion_minimum_are_raised(self):
        calls: dict = {}
        champions = [
            {
                "agent_id": index,
                "agent_type": "human",
                "health": 100,
                "speed": 3,
                "is_super": False,
                "weapon_tier": 3,
                "strength": None,
            }
            for index in (1, 2, 3)
        ]

        class FakeDB:
            def __init__(self, _config, map_bounds=None):
                pass

            def connect(self):
                pass

            def try_run_lock(self):
                return True

            def load_champions(self):
                return champions

            def save_champion_upgrades(self, rows):
                calls["promoted"] = rows

            def insert_agents(self, specs):
                calls["spawned"] = specs

            def mark_champions(self):
                calls["marked"] = True

            def close(self):
                pass

        class FakeReport:
            def render(self):
                return "report"

        class FakeSimulation:
            def __init__(self, _db, **_kwargs):
                pass

            def prewarm(self, count):
                calls["prewarm"] = count

            def run(self):
                return FakeReport()

            def close(self):
                pass

        def fake_spawns(humans, zombies, _rng, map_bounds=None):
            calls["new"] = (humans, zombies)
            return []

        args = argparse.Namespace(
            run=True,
            humans=1,
            zombies=0,
            seed=1,
            mode="headless",
            headless=True,
            workers=1,
            backend="serial",
            sectors=None,
            event_shards=None,
            carry_champions=True,
        )
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main, "ensure_database", return_value=True)
            )
            stack.enter_context(
                patch.object(main, "load_champions", return_value=champions)
            )
            stack.enter_context(patch.object(main, "DBManager", FakeDB))
            stack.enter_context(
                patch.object(main, "try_bootstrap", return_value=True)
            )
            stack.enter_context(
                patch.object(main.schema, "clear_agents", return_value=0)
            )
            stack.enter_context(
                patch.object(main, "generate_spawns", side_effect=fake_spawns)
            )
            stack.enter_context(
                patch.object(main, "Simulation", FakeSimulation)
            )
            out = stack.enter_context(redirect_stdout(io.StringIO()))
            ok = main.handle_simulation({}, args)
            text = out.getvalue()

        self.assertTrue(ok)
        self.assertIn("Raised to the champion minimum: 3 human(s)", text)
        self.assertEqual(calls["new"], (0, 0))
        self.assertEqual(calls["prewarm"], 3)
        self.assertEqual(len(calls["promoted"]), 3)


class StaleChampionsTests(unittest.TestCase):
    def test_champions_changed_after_the_prompt_abort_before_clearing(self):
        champions = [{"agent_id": 1, "agent_type": "human"}]

        class FakeDB:
            def __init__(self, _config, map_bounds=None):
                pass

            def connect(self):
                pass

            def try_run_lock(self):
                return True

            def load_champions(self):
                return []  # another process cleared them after the prompt

            def close(self):
                pass

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main, "ensure_database", return_value=True)
            )
            stack.enter_context(
                patch.object(
                    main, "load_champion_counts", return_value={"human": 1}
                )
            )
            stack.enter_context(
                patch.object(main, "load_champions", return_value=champions)
            )
            stack.enter_context(patch.object(main, "ask_yes_no", return_value=True))
            stack.enter_context(patch.object(main, "ask_int", side_effect=[5, 5]))
            stack.enter_context(patch.object(main, "ask_choice", return_value=2))
            stack.enter_context(patch("builtins.input", return_value=""))
            stack.enter_context(patch.object(main, "DBManager", FakeDB))
            stack.enter_context(
                patch.object(main, "try_bootstrap", return_value=True)
            )
            clear = stack.enter_context(
                patch.object(main.schema, "clear_agents")
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertFalse(main.handle_simulation({}, None))

        clear.assert_not_called()

    def test_declining_after_the_champion_count_changes_aborts_before_clearing(self):
        class FakeDB:
            def __init__(self, _config, map_bounds=None):
                pass

            def connect(self):
                pass

            def try_run_lock(self):
                return True

            def count_champions(self):
                return {"human": 5}  # another run finished after the prompt

            def close(self):
                pass

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(main, "ensure_database", return_value=True)
            )
            stack.enter_context(
                patch.object(
                    main, "load_champion_counts", return_value={"human": 2}
                )
            )
            stack.enter_context(
                patch.object(main, "ask_yes_no", return_value=False)
            )
            stack.enter_context(patch.object(main, "ask_int", side_effect=[5, 5]))
            stack.enter_context(patch.object(main, "ask_choice", return_value=2))
            stack.enter_context(patch("builtins.input", return_value=""))
            stack.enter_context(patch.object(main, "DBManager", FakeDB))
            stack.enter_context(
                patch.object(main, "try_bootstrap", return_value=True)
            )
            clear = stack.enter_context(
                patch.object(main.schema, "clear_agents")
            )
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertFalse(main.handle_simulation({}, None))

        clear.assert_not_called()


class LoadChampionsFailureTests(unittest.TestCase):
    def _call(self, error: Exception):
        class FakeDB:
            def __init__(self, _config):
                pass

            def connect(self):
                pass

            def load_champions(self):
                raise error

            def close(self):
                pass

        with patch.object(main, "DBManager", FakeDB):
            return main.load_champions({})

    def test_old_schema_means_no_champions(self):
        error = main.psycopg.errors.UndefinedColumn("is_champion")
        self.assertEqual(self._call(error), [])

    def test_other_database_errors_are_not_hidden(self):
        error = main.psycopg.OperationalError("server went away")
        with self.assertRaises(main.psycopg.Error):
            self._call(error)


if __name__ == "__main__":
    unittest.main()
