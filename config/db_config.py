"""PostgreSQL login configuration.

All login details live in one hand-editable file:

    config/database.ini

Edit that file directly, or use "Database settings" (menu option 3) in the
app, which rewrites it for you.

Resolution order, per value:

    1. ``config/database.ini``   <- the file to edit
    2. environment variables     DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    3. built-in defaults

The keys follow libpq naming (``dbname``, not ``database``) because that is
what psycopg 3 passes straight through to the PostgreSQL client library.
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "database.ini"
SECTION = "postgresql"

# Keep a wrong host/port from freezing the app. libpq waits forever by default.
CONNECT_TIMEOUT = 5  # seconds

DEFAULT_DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "zombie_sim",
    "user": "postgres",
    "password": "postgres",
}

ENV_VARS = {
    "host": "DB_HOST",
    "port": "DB_PORT",
    "dbname": "DB_NAME",
    "user": "DB_USER",
    "password": "DB_PASSWORD",
}

INI_HEADER = """\
# ===========================================================================
#  PostgreSQL login  --  EDIT THIS FILE
# ===========================================================================
#
#  This is the only place the database login is configured.
#  Fill in the details for your PostgreSQL server, save the file, then run:
#
#      .\\venv\\Scripts\\python.exe main.py
#
#  You do NOT need to create the database or tables by hand - they are
#  created automatically on first run. The "dbname" below only needs to be
#  a name your user is allowed to create.
#
#  Leaving a value blank falls back to the matching environment variable
#  (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD) and then to a built-in
#  default, so you can also configure this from your shell if you prefer.
#
#  The password is stored here in plain text. This file is personal to this
#  machine - do not commit it to a public repository.
#
#  Tip: inside the app, menu option 3 ("Database settings") edits this file
#  for you, including the hidden password prompt.
# ===========================================================================

"""


def _render_ini(values: dict) -> str:
    """Build the INI text, keeping the explanatory comments on every rewrite."""
    lines = [INI_HEADER.rstrip("\n"), "", f"[{SECTION}]"]
    for key in ("host", "port", "dbname", "user", "password"):
        lines.append(f"{key} = {values.get(key, '')}")
    return "\n".join(lines) + "\n"


def _read_ini() -> dict:
    """Return the ``[postgresql]`` section as a dict, or ``{}`` if unusable."""
    if not CONFIG_PATH.exists():
        return {}

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(CONFIG_PATH, encoding="utf-8")
    except (OSError, configparser.Error):
        # A broken file must not stop the app from starting.
        return {}

    if not parser.has_section(SECTION):
        return {}
    return {key: value.strip() for key, value in parser.items(SECTION)}


def ensure_config_file() -> Path:
    """Create ``config/database.ini`` with commented defaults if missing."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(_render_ini(DEFAULT_DB_CONFIG), encoding="utf-8")
    return CONFIG_PATH


def load_db_config() -> dict:
    """Return the effective connection settings."""
    config = dict(DEFAULT_DB_CONFIG)

    # 2. environment variables fill in anything the file does not provide
    for key, env_name in ENV_VARS.items():
        raw = os.getenv(env_name)
        if raw not in (None, ""):
            config[key] = raw

    # 1. the INI file wins over both
    ini = _read_ini()
    for key, value in ini.items():
        if key not in config:
            continue
        if value != "":
            config[key] = value
        elif key == "password":
            # An explicit blank means trust auth / .pgpass, never the built-in
            # default. The environment still gets a say above.
            config[key] = os.getenv("DB_PASSWORD") or ""

    try:
        config["port"] = int(config["port"])
    except (TypeError, ValueError):
        config["port"] = DEFAULT_DB_CONFIG["port"]

    return config


def save_db_config(config: dict) -> Path:
    """Write the settings back to ``config/database.ini``."""
    values = {key: config.get(key, DEFAULT_DB_CONFIG[key]) for key in DEFAULT_DB_CONFIG}
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(_render_ini(values), encoding="utf-8")
    return CONFIG_PATH


def connect_params(config: dict, *, dbname: str | None = None) -> dict:
    """Keyword arguments for ``psycopg.connect``.

    psycopg 3 / libpq expect ``dbname``. Only the five known keys are passed
    on, so a stray entry in the INI file can never break the connection. An
    empty password is omitted so trust auth and ``.pgpass`` still work.
    """
    params = {
        "host": config["host"],
        "port": int(config["port"]),
        "dbname": dbname or config["dbname"],
        "user": config["user"],
        "connect_timeout": CONNECT_TIMEOUT,
    }
    password = config.get("password") or ""
    if password:
        params["password"] = password
    return params


def describe_db_config(config: dict) -> str:
    """Short, password-free summary for the menu header."""
    return f"{config['user']}@{config['host']}:{config['port']}/{config['dbname']}"
