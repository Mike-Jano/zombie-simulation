"""TKinter front end for the zombie simulation.

``main.py`` opens this instead of the terminal menu; ``main.py --menu`` still
shows the old menu. The run itself is untouched: the Start button builds the
same ``argparse.Namespace`` that ``--run`` uses and calls
``main.handle_simulation`` in a worker thread, so the terminal keeps printing
the same progress lines and report. "Database settings..." mirrors menu
option 3: it tests the login, then rewrites ``config/database.ini``.

All Tk calls stay on the main thread; the worker posts messages to a queue that
``root.after`` drains.
"""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
import traceback
from tkinter import messagebox, scrolledtext, ttk

import psycopg

try:
    import sv_ttk
except ImportError:  # styling only; the form still works on the default theme
    sv_ttk = None

from config import simulation_config as cfg
from config.db_config import describe_db_config, save_db_config
from runtime.capabilities import VALID_BACKENDS

ACCEL_CHOICES = ("auto", "off", "numpy")
MODES = ("html", "headless")


# --------------------------------------------------------------------------- #
# Input model (no Tk, unit-tested by tests/test_gui.py)
# --------------------------------------------------------------------------- #
def _int_field(raw: str, label: str, minimum: int, maximum: int | None = None) -> int:
    text = raw.strip()
    if not text:
        raise ValueError(f"{label} is required.")
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{label} must be a whole number.") from None
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be at most {maximum}.")
    return value


def _optional_int(raw: str, label: str, minimum: int | None = None) -> int | None:
    text = raw.strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError:
        raise ValueError(f"{label} must be a whole number.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def build_run_args(values: dict) -> argparse.Namespace:
    """Turn the form's strings into the Namespace ``--run`` expects."""
    humans = _int_field(values["humans"], "Humans", 0, cfg.MAX_AGENTS)
    zombies = _int_field(values["zombies"], "Zombies", 0, cfg.MAX_AGENTS)
    if humans + zombies == 0:
        raise ValueError("At least one agent is required.")
    if humans + zombies > cfg.MAX_AGENTS:
        raise ValueError(f"Too many agents: the limit is {cfg.MAX_AGENTS}.")

    mode = values.get("mode", "html")
    if mode not in MODES:
        raise ValueError("Visualisation must be HTML or headless.")

    backend = values.get("backend", "").strip() or None
    if backend is not None and backend not in VALID_BACKENDS:
        raise ValueError(
            f"Backend must be one of {', '.join(VALID_BACKENDS)} or blank for auto."
        )

    accel = values.get("accel", "").strip() or None
    if accel is not None and accel not in ACCEL_CHOICES:
        raise ValueError(
            f"Accel must be one of {', '.join(ACCEL_CHOICES)} or blank for auto."
        )

    return argparse.Namespace(
        run=True,
        humans=humans,
        zombies=zombies,
        seed=_optional_int(values.get("seed", ""), "Seed"),
        mode=mode,
        headless=mode == "headless",
        carry_champions=bool(values.get("carry", False)),
        workers=_optional_int(values.get("workers", ""), "Workers", 0),
        backend=backend,
        sectors=_optional_int(values.get("sectors", ""), "Sectors", 0),
        event_shards=_optional_int(
            values.get("event_shards", ""), "Event shards", 0
        ),
        accel=accel,
    )


def champion_message(human: int, zombie: int) -> str:
    """The "Previous run" line, with singular/plural handled."""
    def count(n: int, noun: str) -> str:
        return f"{n} {noun}" if n == 1 else f"{n} {noun}s"

    if not human and not zombie:
        return "No champions from the previous run."
    return (
        f"Previous run: {count(human, 'human')}, "
        f"{count(zombie, 'zombie')} survived."
    )


DB_TEXT_FIELDS = (
    ("host", "Host"),
    ("port", "Port"),
    ("dbname", "Database name"),
    ("user", "User"),
)


def build_db_settings(values: dict, current: dict) -> dict:
    """Apply the terminal "Database settings" rules to the form's strings.

    A blank text field keeps the current value, ``-`` clears the password, and
    a blank password keeps it. ``current`` is copied, never modified.
    """
    updated = dict(current)
    for key, label in DB_TEXT_FIELDS:
        raw = (values.get(key) or "").strip()
        if not raw:
            continue
        if key == "port":
            try:
                port = int(raw)
            except ValueError:
                raise ValueError(f"{label} must be a whole number.") from None
            if not 1 <= port <= 65535:
                raise ValueError(f"{label} must be between 1 and 65535.")
            updated[key] = port
        else:
            updated[key] = raw

    password = values.get("password") or ""
    if password == "-":
        updated["password"] = ""
    elif password:
        updated["password"] = password
    return updated


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
class _App:
    def __init__(
        self,
        root: tk.Tk,
        config: dict,
        handle_simulation,
        handle_exit,
        load_champion_counts,
        check_database,
    ) -> None:
        self.root = root
        self.config = config
        self.handle_simulation = handle_simulation
        self.handle_exit = handle_exit
        self.load_champion_counts = load_champion_counts
        self.check_database = check_database

        self.running = False
        self.report = None
        self.events: queue.Queue | None = None
        self.progressbar: ttk.Progressbar | None = None
        self.closed = False

        self.settings_window: tk.Toplevel | None = None
        self.settings_values: dict[str, tk.StringVar] = {}
        self.settings_status: tk.StringVar | None = None
        self.settings_save: ttk.Button | None = None
        self.settings_queue: queue.Queue = queue.Queue()

        # The champions line is filled by a background read so a dead database
        # cannot freeze the form. Start refuses until the read settles.
        self.champion_counts: dict | None = None
        self.champion_error: str | None = None
        self.champion_queue: queue.Queue = queue.Queue()
        self._champion_token = 0
        self._champion_polling = False

        self.start = ttk.Frame(root, padding=14)
        self.simulating = ttk.Frame(root, padding=14)
        self.summary = ttk.Frame(root, padding=14)

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.bind("<Return>", self._on_return)
        self._show_start()

    # -- shared ------------------------------------------------------------
    def _show(self, frame: ttk.Frame) -> None:
        for other in (self.start, self.simulating, self.summary):
            other.pack_forget()
        frame.pack(fill="both", expand=True)

    @staticmethod
    def _clear(frame: ttk.Frame) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _read_champion_counts(self) -> None:
        """Refresh the champions line off the Tk main thread.

        The result is remembered: Start hands these counts to
        ``handle_simulation``, which re-checks them under the run lock before
        anything is cleared.
        """
        self._champion_token += 1
        token = self._champion_token
        self.champion_counts = None
        self.champion_error = None
        self.champion_text.set("Reading champions...")
        self.carry.set(False)
        self.carry_box.state(["disabled"])
        self.db_status.set("Checking...")
        self.db_status_label.configure(foreground="#6b6b6b")

        def worker() -> None:
            try:
                counts = self.load_champion_counts(self.config)
            except psycopg.Error as exc:
                self.champion_queue.put((token, None, str(exc)))
            except Exception as exc:  # the form must not wait forever
                traceback.print_exc()
                self.champion_queue.put((token, None, str(exc)))
            else:
                self.champion_queue.put((token, counts, None))

        threading.Thread(target=worker, daemon=True).start()
        if not self._champion_polling:
            self._champion_polling = True
            self.root.after(50, self._poll_champion_counts)

    def _poll_champion_counts(self) -> None:
        if self.closed:
            return
        try:
            while True:
                token, counts, error = self.champion_queue.get_nowait()
                if token == self._champion_token:
                    self._apply_champion_counts(counts, error)
        except queue.Empty:
            pass
        if self.champion_counts is None and self.champion_error is None:
            self.root.after(50, self._poll_champion_counts)
        else:
            self._champion_polling = False

    def _apply_champion_counts(self, counts: dict | None, error: str | None) -> None:
        self.champion_counts = counts
        self.champion_error = error
        if counts is None:
            self.champion_text.set(f"Champions could not be read: {error}")
            self.carry.set(False)
            self.carry_box.state(["disabled"])
            self.db_status.set("Not connected")
            self.db_status_label.configure(foreground="#b00020")
        else:
            self.db_status.set("Connected")
            self.db_status_label.configure(foreground="#107c10")
            human = counts.get("human", 0)
            zombie = counts.get("zombie", 0)
            self.champion_text.set(champion_message(human, zombie))
            if human or zombie:
                self.carry_box.state(["!disabled"])
            else:
                self.carry.set(False)
                self.carry_box.state(["disabled"])
        # The message may wrap to another line; keep the floor in step.
        self._fit_min_size()

    # -- start screen ------------------------------------------------------
    def _show_start(self) -> None:
        if self.progressbar is not None:
            self.progressbar.stop()
            self.progressbar = None
        self.root.title("Zombie Simulation")
        self._clear(self.start)

        self.humans = tk.StringVar(value="40")
        self.zombies = tk.StringVar(value="40")
        self.seed = tk.StringVar()
        self.mode = tk.StringVar(value="html")
        self.carry = tk.BooleanVar(value=False)
        self.workers = tk.StringVar()
        self.backend = tk.StringVar()
        self.sectors = tk.StringVar()
        self.event_shards = tk.StringVar()
        self.accel = tk.StringVar()
        self.error = tk.StringVar()
        self.advanced = tk.BooleanVar(value=False)

        frame = self.start
        frame.columnconfigure(0, minsize=96)
        frame.columnconfigure(1, weight=1)
        muted = ("Segoe UI", 8)
        pad = 8

        ttk.Label(
            frame, text="Zombie Simulation", font=("Segoe UI", 15, "bold")
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=pad)
        ttk.Label(
            frame, text="Configure and run a simulation", foreground="#6b6b6b"
        ).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=pad, pady=(0, 6)
        )

        def group(title: str, row: int) -> ttk.Frame:
            box = ttk.LabelFrame(frame, text=title, padding=(10, 2))
            box.grid(
                row=row, column=0, columnspan=2, sticky="ew",
                padx=pad, pady=(0, 6),
            )
            box.columnconfigure(0, minsize=96)
            box.columnconfigure(1, weight=1)
            return box

        def field(
            parent: ttk.Frame,
            row: int,
            label: str,
            variable: tk.StringVar,
            hint: str = "",
            spin: bool = False,
        ) -> int:
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
            box = ttk.Frame(parent)
            box.grid(row=row, column=1, sticky="ew", pady=2)
            if spin:
                widget: ttk.Widget = ttk.Spinbox(
                    box,
                    textvariable=variable,
                    from_=0,
                    to=cfg.MAX_AGENTS,
                    increment=10,
                    wrap=False,
                    width=10,
                )
            else:
                widget = ttk.Entry(box, textvariable=variable, width=16)
            widget.pack(side="left")
            if hint:
                ttk.Label(box, text=hint, foreground="#6b6b6b", font=muted).pack(
                    side="left", padx=(8, 0)
                )
            return row + 1

        previous = group("Previous run", 2)
        self.champion_text = tk.StringVar()
        ttk.Label(
            previous,
            textvariable=self.champion_text,
            wraplength=340,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        self.carry_box = ttk.Checkbutton(
            previous,
            text="Carry the champions into this run",
            variable=self.carry,
        )
        self.carry_box.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        population = group("Population", 3)
        population_row = field(population, 0, "Humans", self.humans, spin=True)
        population_row = field(
            population, population_row, "Zombies", self.zombies, spin=True
        )
        field(population, population_row, "Random seed", self.seed, "blank = random")

        output = group("Output", 4)
        ttk.Radiobutton(
            output,
            text="HTML file (opens in your browser)",
            value="html",
            variable=self.mode,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Radiobutton(
            output,
            text="Headless (summary only)",
            value="headless",
            variable=self.mode,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=2)

        self.advanced_text = tk.StringVar()
        ttk.Checkbutton(
            frame,
            textvariable=self.advanced_text,
            variable=self.advanced,
            command=self._toggle_advanced,
            style="Toolbutton",
        ).grid(
            # 4 + the theme's 8px button padding lines the text up with the
            # LabelFrame titles.
            row=5, column=0, columnspan=2, sticky="w", padx=4
        )

        self.advanced_frame = ttk.Frame(frame)
        self.advanced_frame.grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=(pad + 12, pad)
        )
        self.advanced_frame.columnconfigure(0, minsize=96)
        self.advanced_frame.columnconfigure(1, weight=1)
        advanced_row = 0

        def advanced_combo(label: str, variable: tk.StringVar, choices) -> None:
            nonlocal advanced_row
            ttk.Label(self.advanced_frame, text=label).grid(
                row=advanced_row, column=0, sticky="w", pady=2
            )
            ttk.Combobox(
                self.advanced_frame,
                textvariable=variable,
                values=choices,
                state="readonly",
                width=14,
            ).grid(row=advanced_row, column=1, sticky="w", padx=(12, 0), pady=2)
            advanced_row += 1

        advanced_row = field(
            self.advanced_frame,
            advanced_row,
            "Workers",
            self.workers,
            "blank = auto (one per CPU)",
        )
        advanced_combo("Backend", self.backend, ("", *VALID_BACKENDS))
        advanced_row = field(
            self.advanced_frame, advanced_row, "Sectors", self.sectors, "blank = auto"
        )
        advanced_row = field(
            self.advanced_frame,
            advanced_row,
            "Event shards",
            self.event_shards,
            "blank = auto",
        )
        advanced_combo("Accel", self.accel, ("", *ACCEL_CHOICES))

        ttk.Label(
            frame,
            textvariable=self.error,
            foreground="#b00020",
            wraplength=340,
            justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="w", padx=pad, pady=(8, 0))

        actions = ttk.Frame(frame)
        actions.grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=pad, pady=(10, 0)
        )
        ttk.Button(
            actions, text="Database settings...", command=self._open_db_settings
        ).pack(side="left")
        ttk.Button(
            actions,
            text="Start",
            command=self._start,
            style="Accent.TButton",
            width=12,
        ).pack(side="right")

        status_row = ttk.Frame(frame)
        status_row.grid(
            row=9, column=0, columnspan=2, sticky="ew", padx=pad, pady=(8, 0)
        )
        status_row.columnconfigure(1, weight=1)
        ttk.Label(
            status_row, text="Database:", foreground="#6b6b6b", font=muted
        ).grid(row=0, column=0, sticky="w")
        self.db_text = tk.StringVar(value=describe_db_config(self.config))
        ttk.Label(
            status_row, textvariable=self.db_text, foreground="#6b6b6b", font=muted
        ).grid(row=0, column=1, sticky="w", padx=(4, 8))
        self.db_status = tk.StringVar()
        self.db_status_label = ttk.Label(
            status_row, textvariable=self.db_status, font=muted
        )
        self.db_status_label.grid(row=0, column=2, sticky="e")

        self._toggle_advanced()
        self._show(self.start)
        self.root.geometry("")
        self._fit_min_size()
        self._read_champion_counts()

    def _fit_min_size(self) -> None:
        """Stop the window shrinking below the form that is on screen."""
        self.root.update_idletasks()
        self.root.minsize(
            self.start.winfo_reqwidth(), self.start.winfo_reqheight()
        )

    def _toggle_advanced(self) -> None:
        expanded = self.advanced.get()
        self.advanced_text.set(
            ("\u25be " if expanded else "\u25b8 ") + "Advanced options"
        )
        if expanded:
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()
        # Let the window grow/shrink to the form instead of clipping the extra
        # rows (the summary screen sets its own geometry).
        if self.start.winfo_ismapped():
            self.root.geometry("")
            self._fit_min_size()

    # -- database settings -------------------------------------------------
    def _open_db_settings(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Database settings")
        win.transient(self.root)

        frame = ttk.Frame(win, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        self.settings_values = {
            "host": tk.StringVar(value=str(self.config["host"])),
            "port": tk.StringVar(value=str(self.config["port"])),
            "dbname": tk.StringVar(value=str(self.config["dbname"])),
            "user": tk.StringVar(value=str(self.config["user"])),
            "password": tk.StringVar(),
        }
        row = 0
        for key, label in DB_TEXT_FIELDS:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=2)
            ttk.Entry(frame, textvariable=self.settings_values[key], width=28).grid(
                row=row, column=1, sticky="ew", padx=(12, 0), pady=2
            )
            row += 1
        ttk.Label(frame, text="Password").grid(row=row, column=0, sticky="w", pady=2)
        ttk.Entry(
            frame, textvariable=self.settings_values["password"], show="*", width=28
        ).grid(row=row, column=1, sticky="ew", padx=(12, 0), pady=2)
        row += 1
        ttk.Label(
            frame,
            text="Blank keeps the current value; enter '-' to clear the password.",
            foreground="#666666",
        ).grid(row=row, column=0, columnspan=2, sticky="w")
        row += 1

        self.settings_status = tk.StringVar()
        ttk.Label(
            frame,
            textvariable=self.settings_status,
            foreground="#b00020",
            wraplength=340,
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row += 1

        buttons = ttk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=2, sticky="e", pady=(10, 0))
        self.settings_save = ttk.Button(
            buttons, text="Save and test", command=self._save_db_settings
        )
        self.settings_save.pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side="left")

        self.settings_window = win
        win.grab_set()

    def _save_db_settings(self) -> None:
        try:
            updated = build_db_settings(
                {key: var.get() for key, var in self.settings_values.items()},
                self.config,
            )
        except ValueError as exc:
            self.settings_status.set(str(exc))
            return

        win = self.settings_window
        self.settings_status.set("Testing the connection...")
        self.settings_save.state(["disabled"])

        def worker() -> None:
            try:
                result = self.check_database(updated)
            except Exception as exc:  # thread boundary: always answer the window
                traceback.print_exc()
                result = (False, str(exc))
            self.settings_queue.put((win, updated, result))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll_settings)

    def _poll_settings(self) -> None:
        if self.closed:
            return
        try:
            win, updated, (ok, missing) = self.settings_queue.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_settings)
            return
        self._apply_db_settings(win, updated, ok, missing)

    def _apply_db_settings(
        self, win: tk.Toplevel, updated: dict, ok: bool, missing: str | None
    ) -> None:
        if not win.winfo_exists():
            return  # the dialog was cancelled while the test ran
        if not ok:
            self.settings_status.set(
                "Not saved - the connection test failed. "
                "The terminal shows the PostgreSQL error."
            )
            self.settings_save.state(["!disabled"])
            return

        try:
            saved_to = save_db_config(updated)
        except OSError as exc:
            self.settings_status.set(f"Could not write the settings file: {exc}")
            self.settings_save.state(["!disabled"])
            return

        # The run and exit callbacks hold this same dict, so update it in place.
        self.config.clear()
        self.config.update(updated)
        self.db_text.set(describe_db_config(self.config))
        win.destroy()
        # Champions live in the newly configured database now.
        self._read_champion_counts()
        if missing:
            messagebox.showinfo(
                "Database settings",
                f"Saved to {saved_to}\n\n{missing}",
            )

    def _on_return(self, _event: tk.Event) -> None:
        # Enter starts the run, but not while it is running, not under the
        # modal database dialog, and not on the summary screen.
        if self.running or self.root.grab_current() is not None:
            return
        if self.start.winfo_manager() == "pack":
            self._start()

    def _show_error(self, message: str) -> None:
        self.error.set(message)
        self._fit_min_size()

    def _start(self) -> None:
        if self.champion_counts is None:
            if self.champion_error is not None:
                self._show_error(
                    "Start refused: the champions could not be read "
                    f"({self.champion_error})."
                )
            else:
                self._show_error("Still reading the champions - try again.")
            return
        values = {
            "humans": self.humans.get(),
            "zombies": self.zombies.get(),
            "seed": self.seed.get(),
            "mode": self.mode.get(),
            "carry": self.carry.get(),
            "workers": self.workers.get(),
            "backend": self.backend.get(),
            "sectors": self.sectors.get(),
            "event_shards": self.event_shards.get(),
            "accel": self.accel.get(),
        }
        try:
            args = build_run_args(values)
        except ValueError as exc:
            self._show_error(str(exc))
            return

        self.report = None
        self.events = queue.Queue()
        self._show_simulating()
        threading.Thread(
            target=self._run_worker,
            args=(args, self.champion_counts),
            daemon=True,
        ).start()
        self.root.after(100, self._poll)

    # -- run ---------------------------------------------------------------
    def _run_worker(
        self, args: argparse.Namespace, champion_counts: dict
    ) -> None:
        events = self.events

        def progress(line: str) -> None:
            events.put(("progress", line))

        def report(completed) -> None:
            events.put(("report", completed))

        try:
            ok = self.handle_simulation(
                self.config,
                args,
                progress_cb=progress,
                report_cb=report,
                champion_counts=champion_counts,
            )
        except Exception as exc:  # thread boundary: always tell the window
            traceback.print_exc()
            if isinstance(exc, psycopg.Error):
                print()
                print("Could not reach PostgreSQL:")
                print(f"  {str(exc).strip()}")
                print(
                    "Check that the server is running and the settings are "
                    "correct, then try again."
                )
            events.put(("error", str(exc)))
        else:
            events.put(("done", ok))

    def _show_simulating(self) -> None:
        self._clear(self.simulating)
        ttk.Label(
            self.simulating, text="Simulating...", font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        self.progressbar = ttk.Progressbar(
            self.simulating, mode="indeterminate", length=440
        )
        self.progressbar.pack(anchor="w", pady=12)
        self.progressbar.start(15)
        self.status = tk.StringVar(value="Starting...")
        ttk.Label(
            self.simulating, textvariable=self.status, wraplength=440,
            justify="left",
        ).pack(anchor="w")
        self.running = True
        self._show(self.simulating)
        self.root.minsize(1, 1)
        self.root.geometry("500x150")

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    self.status.set(payload)
                elif kind == "report":
                    self.report = payload
                elif kind == "done":
                    self._finish(payload)
                    return
                else:  # "error"
                    self._fail(payload)
                    return
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _finish(self, ok: bool) -> None:
        self.running = False
        if self.progressbar is not None:
            self.progressbar.stop()
        if ok and self.report is not None:
            self._show_summary()
            return
        if not ok:
            messagebox.showwarning(
                "Run not started",
                "The simulation did not run. The terminal shows why.",
            )
        self._show_start()

    def _fail(self, message: str) -> None:
        self.running = False
        if self.progressbar is not None:
            self.progressbar.stop()
        messagebox.showerror("Simulation failed", message)
        self._show_start()

    # -- summary -----------------------------------------------------------
    def _show_summary(self) -> None:
        self.root.title("Zombie Simulation - summary")
        self._clear(self.summary)

        ttk.Label(
            self.summary, text="Simulation summary", font=("Segoe UI", 13, "bold")
        ).pack(anchor="w")
        text = scrolledtext.ScrolledText(
            self.summary, wrap="none", font=("Consolas", 9), width=100, height=26
        )
        text.pack(fill="both", expand=True, pady=10)
        text.insert("1.0", self.report.render())
        text.configure(state="disabled")

        buttons = ttk.Frame(self.summary)
        buttons.pack(fill="x")
        ttk.Button(
            buttons, text="New simulation", command=self._show_start
        ).pack(side="left")
        ttk.Button(
            buttons, text="Close program", command=self._close
        ).pack(side="right")

        self._show(self.summary)
        self.root.minsize(480, 320)
        self.root.geometry("820x660")

    # -- closing -----------------------------------------------------------
    def _on_close(self) -> None:
        if self.running:
            messagebox.showinfo(
                "Simulation running",
                "Wait for the run to finish - closing now would leave the "
                "database half-written.",
            )
            return
        self._close()

    def _close(self) -> None:
        self.closed = True
        self.handle_exit(self.config)
        self.root.destroy()


def run_gui(
    config: dict,
    *,
    handle_simulation,
    handle_exit,
    load_champion_counts,
    check_database,
) -> int:
    root = tk.Tk()
    root.title("Zombie Simulation")
    if sv_ttk is not None:
        sv_ttk.set_theme("light")
    else:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    _App(
        root,
        config,
        handle_simulation,
        handle_exit,
        load_champion_counts,
        check_database,
    )
    root.mainloop()
    return 0
