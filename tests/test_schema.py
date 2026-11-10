"""``ensure_database``: reuse the target DB, create it only when missing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db import schema  # noqa: E402

CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "zombie_sim",
    "user": "postgres",
    "password": "postgres",
}


class EnsureDatabaseTests(unittest.TestCase):
    def test_existing_database_is_used_directly(self):
        target = MagicMock()
        with patch.object(schema.psycopg, "connect", return_value=target) as connect:
            schema.ensure_database(CONFIG)

        connect.assert_called_once()
        self.assertEqual(connect.call_args.kwargs["dbname"], "zombie_sim")
        target.close.assert_called_once()

    def test_missing_database_is_created_from_the_maintenance_connection(self):
        maintenance = MagicMock()
        cur = maintenance.cursor.return_value.__enter__.return_value
        cur.fetchone.return_value = None  # the catalog says it is missing
        missing = schema.psycopg.OperationalError("database does not exist")

        with patch.object(
            schema.psycopg, "connect", side_effect=[missing, maintenance]
        ) as connect:
            schema.ensure_database(CONFIG)

        self.assertEqual(connect.call_count, 2)
        self.assertEqual(connect.call_args.kwargs["dbname"], "postgres")
        # Existence check plus CREATE DATABASE.
        self.assertEqual(cur.execute.call_count, 2)
        maintenance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
