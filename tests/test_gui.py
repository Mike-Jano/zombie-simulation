"""Headless tests for the GUI form -> Namespace conversion (no window)."""

from __future__ import annotations

import sys
import time
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

import psycopg

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gui  # noqa: E402


def values(**overrides) -> dict:
    base = {
        "humans": "40",
        "zombies": "40",
        "seed": "",
        "mode": "html",
        "carry": False,
        "workers": "",
        "backend": "",
        "sectors": "",
        "event_shards": "",
        "accel": "",
    }
    base.update(overrides)
    return base


class BuildRunArgsTests(unittest.TestCase):
    def test_defaults_match_the_terminal(self):
        args = gui.build_run_args(values())
        self.assertTrue(args.run)
        self.assertEqual((args.humans, args.zombies), (40, 40))
        self.assertEqual(args.mode, "html")
        self.assertFalse(args.headless)
        self.assertIsNone(args.seed)
        self.assertIsNone(args.workers)
        self.assertIsNone(args.backend)
        self.assertIsNone(args.sectors)
        self.assertIsNone(args.event_shards)
        self.assertIsNone(args.accel)
        self.assertFalse(args.carry_champions)

    def test_headless_sets_both_mode_fields(self):
        args = gui.build_run_args(values(mode="headless"))
        self.assertEqual(args.mode, "headless")
        self.assertTrue(args.headless)

    def test_advanced_values_are_parsed(self):
        args = gui.build_run_args(
            values(
                seed="-7",
                carry=True,
                workers="4",
                backend="thread",
                sectors="2",
                event_shards="3",
                accel="numpy",
            )
        )
        self.assertEqual(args.seed, -7)
        self.assertTrue(args.carry_champions)
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.backend, "thread")
        self.assertEqual(args.sectors, 2)
        self.assertEqual(args.event_shards, 3)
        self.assertEqual(args.accel, "numpy")

    def test_blank_humans_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Humans is required"):
            gui.build_run_args(values(humans=" "))

    def test_non_integer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "whole number"):
            gui.build_run_args(values(zombies="many"))

    def test_negative_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 0"):
            gui.build_run_args(values(humans="-1"))

    def test_per_side_limit_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "at most"):
            gui.build_run_args(values(humans=str(gui.cfg.MAX_AGENTS + 1)))

    def test_combined_limit_is_enforced(self):
        half = gui.cfg.MAX_AGENTS // 2 + 1
        with self.assertRaisesRegex(ValueError, "Too many agents"):
            gui.build_run_args(values(humans=str(half), zombies=str(half)))

    def test_zero_agents_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "At least one agent"):
            gui.build_run_args(values(humans="0", zombies="0"))

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Backend must be one of"):
            gui.build_run_args(values(backend="gpu"))

    def test_unknown_accel_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Accel must be one of"):
            gui.build_run_args(values(accel="cuda"))

    def test_negative_workers_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Workers must be at least 0"):
            gui.build_run_args(values(workers="-2"))


class BuildDbSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.current = {
            "host": "localhost",
            "port": 5432,
            "dbname": "zombie_sim",
            "user": "postgres",
            "password": "secret",
        }

    def test_blank_fields_keep_the_current_values(self):
        updated = gui.build_db_settings(
            {"host": "", "port": " ", "dbname": "", "user": "", "password": ""},
            self.current,
        )
        self.assertEqual(updated, self.current)
        self.assertIsNot(updated, self.current)

    def test_values_are_replaced(self):
        updated = gui.build_db_settings(
            {
                "host": "db.example",
                "port": "6543",
                "dbname": "sim2",
                "user": "alice",
                "password": "new",
            },
            self.current,
        )
        self.assertEqual(updated["host"], "db.example")
        self.assertEqual(updated["port"], 6543)
        self.assertEqual(updated["dbname"], "sim2")
        self.assertEqual(updated["user"], "alice")
        self.assertEqual(updated["password"], "new")
        self.assertEqual(self.current["host"], "localhost")

    def test_dash_clears_the_password(self):
        updated = gui.build_db_settings({"password": "-"}, self.current)
        self.assertEqual(updated["password"], "")

    def test_non_integer_port_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "whole number"):
            gui.build_db_settings({"port": "abc"}, self.current)

    def test_out_of_range_port_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 65535"):
            gui.build_db_settings({"port": "70000"}, self.current)


class ChampionMessageTests(unittest.TestCase):
    def test_singular_counts(self):
        self.assertEqual(
            gui.champion_message(1, 1),
            "Previous run: 1 human, 1 zombie survived.",
        )

    def test_plural_with_zero(self):
        self.assertEqual(
            gui.champion_message(16, 0),
            "Previous run: 16 humans, 0 zombies survived.",
        )

    def test_no_champions(self):
        self.assertEqual(
            gui.champion_message(0, 0), "No champions from the previous run."
        )


class FakeReport:
    def render(self) -> str:
        return "summary"

class WindowFlowTests(unittest.TestCase):
    """Start -> worker -> summary, then the summary buttons, on a real root."""

    def setUp(self) -> None:
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"no display: {exc}")
        self.root.withdraw()
        self.config = {
            "host": "localhost",
            "port": 5432,
            "dbname": "zombie_sim",
            "user": "postgres",
            "password": "",
        }
        self.seen: dict = {}
        self.exits: list[dict] = []

    def tearDown(self) -> None:
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _fake_run(
        self,
        config,
        args,
        progress_cb=None,
        report_cb=None,
        champion_counts=None,
    ):
        self.seen["args"] = args
        self.seen["count"] = self.seen.get("count", 0) + 1
        self.seen["counts"] = champion_counts
        if progress_cb:
            progress_cb("round 1")
        if report_cb:
            report_cb(FakeReport())
        return True

    def _fake_exit(self, config):
        self.exits.append(config)

    def _app(
        self, check_database=None, champions=None, load_champions=None
    ) -> gui._App:
        counts = {"human": 2, "zombie": 1} if champions is None else champions
        loader = load_champions or (lambda _config: counts)
        return gui._App(
            self.root,
            self.config,
            self._fake_run,
            self._fake_exit,
            loader,
            check_database or (lambda _config: (True, None)),
        )

    def _pump(self, predicate, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(predicate(), "window did not reach the expected state")

    def _button(self, text: str) -> ttk.Button:
        stack = [self.root]
        while stack:
            widget = stack.pop()
            if isinstance(widget, ttk.Button) and widget.cget("text") == text:
                return widget
            stack.extend(widget.winfo_children())
        self.fail(f"no {text!r} button")

    def test_start_worker_summary_and_both_summary_buttons(self):
        app = self._app()
        self._pump(lambda: app.champion_counts is not None)

        self._button("Start").invoke()
        self._pump(lambda: not app.running)
        self.assertIsNotNone(app.report)
        self.assertEqual(app.summary.winfo_manager(), "pack")
        self.assertEqual(self.root.minsize(), (480, 320))
        self.assertEqual(self.seen["counts"], {"human": 2, "zombie": 1})
        self.assertEqual(
            (self.seen["args"].humans, self.seen["args"].zombies), (40, 40)
        )
        self.assertFalse(self.seen["args"].carry_champions)

        self._button("New simulation").invoke()
        self.root.update()
        self.assertEqual(app.start.winfo_manager(), "pack")
        self.assertEqual(app.summary.winfo_manager(), "")

        self._pump(lambda: app.champion_counts is not None)
        self._button("Start").invoke()
        self._pump(lambda: not app.running)
        self._button("Close program").invoke()
        self.assertEqual(self.exits, [self.config])

    def test_return_starts_a_run_on_the_form_only(self):
        app = self._app()
        self._pump(lambda: app.champion_counts is not None)

        app._on_return(None)
        self._pump(lambda: not app.running)
        self.assertEqual(self.seen["count"], 1)
        self.assertIsNotNone(app.report)

        app._on_return(None)
        self.root.update()
        self.assertEqual(self.seen["count"], 1)
        self.assertEqual(app.summary.winfo_manager(), "pack")

    def test_champion_read_failure_shows_not_connected(self):
        def fail(_config):
            raise psycopg.OperationalError("connection refused")

        app = self._app(load_champions=fail)
        self._pump(lambda: app.champion_error is not None)
        self.assertEqual(app.db_status.get(), "Not connected")
        self.assertIn("disabled", app.carry_box.state())
        self.assertFalse(app.carry.get())
        self.assertIn("could not be read", app.champion_text.get())

    def test_carry_is_enabled_when_champions_exist(self):
        app = self._app()
        self._pump(lambda: app.champion_counts is not None)
        self.assertNotIn("disabled", app.carry_box.state())
        self.assertEqual(
            app.champion_text.get(), "Previous run: 2 humans, 1 zombie survived."
        )
        self.assertEqual(app.db_status.get(), "Connected")

    def test_carry_is_disabled_when_there_are_no_champions(self):
        app = self._app(champions={"human": 0, "zombie": 0})
        self._pump(lambda: app.champion_counts is not None)
        self.assertIn("disabled", app.carry_box.state())
        self.assertFalse(app.carry.get())
        self.assertEqual(
            app.champion_text.get(), "No champions from the previous run."
        )

    def test_database_settings_are_saved_in_place(self):
        app = self._app()
        self._pump(lambda: app.champion_counts is not None)

        self._button("Database settings...").invoke()
        self.root.update()
        app.settings_values["host"].set("newhost")
        app.settings_values["password"].set("-")

        with patch.object(gui, "save_db_config", return_value=Path("x.ini")) as save:
            self._button("Save and test").invoke()
            self._pump(lambda: self.config["host"] == "newhost")

        saved = save.call_args.args[0]
        self.assertEqual(saved["host"], "newhost")
        self.assertEqual(saved["password"], "")
        self.assertEqual(self.config["password"], "")
        self.assertEqual(app.db_text.get(), "postgres@newhost:5432/zombie_sim")
        self.assertFalse(app.settings_window.winfo_exists())

    def test_failed_database_test_keeps_the_settings(self):
        app = self._app(check_database=lambda _config: (False, None))
        self._pump(lambda: app.champion_counts is not None)
        before = dict(self.config)

        self._button("Database settings...").invoke()
        self.root.update()
        app.settings_values["host"].set("badhost")

        with patch.object(gui, "save_db_config") as save:
            self._button("Save and test").invoke()
            self._pump(
                lambda: app.settings_status.get().startswith("Not saved")
            )

        save.assert_not_called()
        self.assertEqual(self.config, before)
        self.assertTrue(app.settings_window.winfo_exists())


if __name__ == "__main__":
    unittest.main()
