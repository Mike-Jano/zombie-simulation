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


if __name__ == "__main__":
    unittest.main()
