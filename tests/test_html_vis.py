"""HTML player frame capture: the cap must keep the closing frame."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import simulation_config as cfg  # noqa: E402
from vis.html_vis import HtmlVisualizer  # noqa: E402


def snapshot(x: float) -> list[tuple]:
    return [(1, "human", x, 0.0, 100, True, 1)]


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


if __name__ == "__main__":
    unittest.main()
