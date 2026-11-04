"""A blank password in the INI file means trust/.pgpass, not the default."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
