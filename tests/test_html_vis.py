"""HTML player frame capture: the cap must keep the closing frame."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from vis.html_vis import HtmlVisualizer, _clear_old_runs  # noqa: E402


def snapshot(x: float) -> list[tuple]:
    return [(1, "human", x, 0.0, 100, True)]


class FrameCapTests(unittest.TestCase):
    def test_closing_frame_replaces_last_when_cap_is_reached(self):
        visualizer = HtmlVisualizer(path="unused.html")
        with patch.object(cfg, "MAX_FRAMES", 2):
            visualizer.frame(snapshot(0.0), {"iteration": 1, "ended": False})
            visualizer.frame(snapshot(1.0), {"iteration": 2, "ended": False})
            visualizer.frame(snapshot(2.0), {"iteration": 3, "ended": True})

        self.assertEqual(len(visualizer.frames), 2)
        self.assertEqual(visualizer.frames[-1]["meta"]["iteration"], 3)
        self.assertTrue(visualizer.frames[-1]["meta"]["ended"])

    def test_record_budget_keeps_a_sample_of_the_whole_run(self):
        visualizer = HtmlVisualizer(path="unused.html")
        with patch.object(cfg, "MAX_FRAME_RECORDS", 10):
            for i in range(20):
                visualizer.frame(snapshot(float(i)), {"iteration": i})

        self.assertLessEqual(visualizer._records, 10)
        self.assertGreater(len(visualizer.frames), 1)
        self.assertEqual(visualizer.frames[-1]["meta"]["iteration"], 19)


class PruneOldRunsTests(unittest.TestCase):
    def test_keeps_new_run_and_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            old = directory / "simulation_old.html"
            new = directory / "simulation_new.html"
            notes = directory / "notes.txt"
            for file in (old, new, notes):
                file.write_text("x", encoding="utf-8")

            _clear_old_runs(directory, keep=new, since=time.time() + 1)

            self.assertFalse(old.exists())
            self.assertTrue(new.exists())
            self.assertTrue(notes.exists())

    def test_keeps_a_player_written_by_another_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            other = directory / "simulation_other.html"
            new = directory / "simulation_new.html"
            other.write_text("other", encoding="utf-8")
            new.write_text("new", encoding="utf-8")

            # This run started before the other process wrote its player.
            _clear_old_runs(directory, keep=new, since=other.stat().st_mtime - 10)

            self.assertTrue(other.exists())
            self.assertTrue(new.exists())

    def test_finish_prunes_previous_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "simulation_old.html"
            old.write_text("old", encoding="utf-8")
            os.utime(old, (time.time() - 60, time.time() - 60))
            with patch.object(cfg, "HTML_OUTPUT_DIR", tmp):
                visualizer = HtmlVisualizer()
                visualizer.start(1, 1)
                visualizer.frame(snapshot(0.0), {"iteration": 1})
                visualizer.finish()

            self.assertFalse(old.exists())
            self.assertTrue(visualizer.path.exists())

    def test_aborted_finish_keeps_previous_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "simulation_old.html"
            old.write_text("old", encoding="utf-8")
            with patch.object(cfg, "HTML_OUTPUT_DIR", tmp):
                visualizer = HtmlVisualizer()
                visualizer.start(1, 1)
                visualizer.frame(snapshot(0.0), {"iteration": 1})
                visualizer.finish(prune=False)

            self.assertTrue(old.exists())
            self.assertTrue(visualizer.path.exists())


class EmptyRunTests(unittest.TestCase):
    def test_finish_without_frames_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cfg, "HTML_OUTPUT_DIR", tmp):
                visualizer = HtmlVisualizer()
                visualizer.start(1, 1)
                visualizer.finish()

            self.assertFalse(visualizer.path.exists())


if __name__ == "__main__":
    unittest.main()
