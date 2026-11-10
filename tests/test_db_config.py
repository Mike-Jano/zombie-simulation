"""A blank password in the INI file means trust/.pgpass, not the default."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import db_config  # noqa: E402


class BlankPasswordTests(unittest.TestCase):
    def test_blank_file_password_uses_no_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "database.ini"
            ini.write_text(
                "[postgresql]\nhost = localhost\nport = 5432\n"
                "dbname = zombie_sim\nuser = postgres\npassword = \n",
                encoding="utf-8",
            )
            with patch.object(db_config, "CONFIG_PATH", ini), patch.dict(
                os.environ, {"DB_PASSWORD": ""}, clear=False
            ):
                config = db_config.load_db_config()
                params = db_config.connect_params(config)

        self.assertEqual(config["password"], "")
        self.assertNotIn("password", params)


class UnusableFileWarningTests(unittest.TestCase):
    def test_missing_section_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "database.ini"
            ini.write_text("[other]\nhost = elsewhere\n", encoding="utf-8")
            err = io.StringIO()
            with patch.object(db_config, "CONFIG_PATH", ini), redirect_stderr(err):
                db_config.load_db_config()

        self.assertIn("no readable [postgresql] section", err.getvalue())

    def test_malformed_file_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "database.ini"
            ini.write_text(
                "[postgresql]\nhost = localhost\nhost = again\n", encoding="utf-8"
            )
            err = io.StringIO()
            with patch.object(db_config, "CONFIG_PATH", ini), redirect_stderr(err):
                self.assertEqual(db_config._read_ini(), {})

        self.assertIn("could not read", err.getvalue())

    def test_malformed_file_warning_does_not_print_the_password(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "database.ini"
            # The password sits above the missing section header, so it is the
            # line configparser would quote in the exception message.
            ini.write_text("password = hunter2\n[postgresql]\n", encoding="utf-8")
            err = io.StringIO()
            with patch.object(db_config, "CONFIG_PATH", ini), redirect_stderr(err):
                self.assertEqual(db_config._read_ini(), {})

        self.assertIn("MissingSectionHeaderError", err.getvalue())
        self.assertNotIn("hunter2", err.getvalue())

    def test_non_utf8_file_warns_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "database.ini"
            ini.write_bytes("[postgresql]\npassword = secret\n".encode("utf-16"))
            err = io.StringIO()
            with patch.object(db_config, "CONFIG_PATH", ini), redirect_stderr(err):
                self.assertEqual(db_config._read_ini(), {})

        self.assertIn("UnicodeDecodeError", err.getvalue())


class Utf8BomTests(unittest.TestCase):
    def test_bom_file_is_read_normally(self):
        with tempfile.TemporaryDirectory() as tmp:
            ini = Path(tmp) / "database.ini"
            ini.write_bytes(b"\xef\xbb\xbf[postgresql]\nhost = h\n")
            with patch.object(db_config, "CONFIG_PATH", ini):
                self.assertEqual(db_config._read_ini(), {"host": "h"})


if __name__ == "__main__":
    unittest.main()
