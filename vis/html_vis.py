"""Self-contained HTML animation of a completed simulation run."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from config import simulation_config as cfg
from utils.sectoring import map_bounds_for

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #141a2e;
    --line: #232a44;
    --text: #e6e9f2;
    --muted: #a2aac2;
    --human: #4d9fff;
    --human-soft: rgba(77,159,255,.18);
    --zombie: #3ddc84;
    --zombie-soft: rgba(61,220,132,.18);
    --accent: #ffd54f;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font-size: 14px;
         font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
  #stage { position: relative; height: clamp(340px, 68vh, 1000px);
           display: flex; align-items: center; justify-content: center;
           background: #0b1020; overflow: hidden; }
  canvas { display: block; width: auto; height: auto; max-width: 100%; max-height: 100%; }
  #hud { position: absolute; top: 12px; left: 12px; z-index: 3;
         display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
         max-width: calc(100% - 24px);
         padding: 6px 12px; font-size: 12px; color: var(--muted);
         background: rgba(11,16,32,.74); border: 1px solid var(--line);
         border-radius: 999px; backdrop-filter: blur(4px); }
  #hud b { color: var(--text); font-variant-numeric: tabular-nums; }
  #hud .dot { width: 8px; height: 8px; margin-right: 4px; }
  #controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
              padding: 12px 24px; background: var(--bg); }
  button { background: #1d2742; color: var(--text); border: 1px solid var(--line);
           border-radius: 8px; padding: 8px 14px; font-size: 14px; cursor: pointer; }
  button:hover { background: #253155; }
  button.primary { background: var(--accent); color: #1a1a1a; border-color: var(--accent); font-weight: 600; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  select { background: #1d2742; color: var(--text); border: 1px solid var(--line);
           border-radius: 8px; padding: 8px; font-size: 14px; }
  #scrub { flex: 1; min-width: 200px; accent-color: var(--accent); }
  #panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 12px; padding: 4px 24px 8px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px; }
  .card h2 { margin: 0 0 10px; font-size: 12px; text-transform: uppercase;
             letter-spacing: .14em; color: var(--muted); font-weight: 600; }
  .stat { display: flex; justify-content: space-between; font-size: 14px; padding: 3px 0; }
  .stat b { font-variant-numeric: tabular-nums; }
  .human { color: var(--human); }
  .zombie { color: var(--zombie); }
  .muted { color: var(--muted); }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 11px;
          border: 1px solid var(--line); letter-spacing: .06em; text-transform: uppercase; }
  .pill.combat { background: rgba(255,107,107,.14); color: #ffb3b3; border-color: rgba(255,107,107,.4); }
  .pill.quiet { background: rgba(77,159,255,.12); color: #a8d2ff; border-color: rgba(77,159,255,.35); }
  #events { list-style: none; margin: 0; padding: 0; font-size: 13px; line-height: 1.5;
            color: var(--muted); max-height: 132px; overflow: auto; }
  #events li { border-bottom: 1px dashed rgba(255,255,255,.06); padding: 3px 0; }
  #events li:last-child { border-bottom: none; }
  .legend { display: flex; gap: 14px; font-size: 13px; color: var(--muted);
            align-items: center; flex-wrap: wrap; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .dot.human { background: var(--human); box-shadow: 0 0 0 3px var(--human-soft); }
  .dot.zombie { background: var(--zombie); box-shadow: 0 0 0 3px var(--zombie-soft); }
  .dot.super { background: transparent; border: 2px solid var(--accent); }
  .dot.sector { background: var(--accent); border-radius: 2px; }
  .dot.ghost { background: linear-gradient(90deg, var(--human) 0 50%, var(--zombie) 50% 100%);
               opacity: .45; }
  .tips { margin: 12px 0 0; font-size: 13px; line-height: 1.55; color: var(--muted); }
  .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
             overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0; }
  footer { padding: 0 24px 24px; color: var(--muted); font-size: 12px; }
  footer p { margin: 0 0 3px; }
  @media (max-width: 700px) {
    #stage { height: clamp(280px, 52vh, 640px); }
    #controls, #panels, footer { padding-left: 14px; padding-right: 14px; }
  }
</style>
</head>
<body>
<h1 class="sr-only">Zombie Simulation</h1>
<main>
<div id="stage">
  <canvas id="map" width="900" height="900" role="img" aria-label="Simulation map"></canvas>
  <div id="hud">
    <span id="h-round">Round -</span>
    <span id="h-sector">Sector -</span>
    <span class="human"><span class="dot human" aria-hidden="true"></span><span class="sr-only">Humans </span><b id="h-humans">-</b></span>
    <span class="zombie"><span class="dot zombie" aria-hidden="true"></span><span class="sr-only">Zombies </span><b id="h-zombies">-</b></span>
    <span id="h-state" class="pill quiet">quiet</span>
  </div>
</div>
<div id="controls">
  <button id="play" class="primary">Pause</button>
  <button id="prev">Prev</button>
  <button id="next">Next</button>
  <button id="restart">Restart</button>
  <input id="scrub" type="range" min="0" max="0" value="0" step="1" aria-label="Timeline">
  <select id="speed" aria-label="Playback speed">
    <option value="0.5">0.5x</option>
    <option value="1">1x</option>
    <option value="2">2x</option>
    <option value="4" selected>4x</option>
    <option value="8">8x</option>
    <option value="16">16x</option>
  </select>
</div>
<section id="panels">
  <div class="card">
    <h2>Status</h2>
    <div class="stat"><span class="muted">Round</span><b id="s-round">-</b></div>
    <div class="stat"><span class="muted">Sector</span><b id="s-sector">-</b></div>
    <div class="stat"><span class="muted">Tick</span><b id="s-tick">-</b></div>
    <div class="stat"><span class="human">Humans alive</span><b id="s-humans">-</b></div>
    <div class="stat"><span class="zombie">Zombies alive</span><b id="s-zombies">-</b></div>
    <div class="stat"><span class="muted">Frame</span><b id="s-frame">-</b></div>
    <div style="margin-top:10px; display:flex; gap:8px; align-items:center;">
      <span id="s-state" class="pill quiet">quiet</span>
      <span id="s-note" class="muted" style="font-size:12px"></span>
    </div>
  </div>
  <div class="card">
    <h2>Events</h2>
    <ul id="events"><li class="muted">No combat this tick.</li></ul>
  </div>
  <div class="card">
    <h2>Key &amp; tips</h2>
    <div class="legend">
      <span><span class="dot human"></span>Human</span>
      <span><span class="dot zombie"></span>Zombie</span>
      <span><span class="dot super"></span>Super (yellow ring)</span>
      <span><span class="dot sector"></span>Active sector</span>
      <span><span class="dot ghost"></span>Recently seen (faded)</span>
    </div>
    <p class="tips">
      Blue dots are humans, green dots are zombies; zombies carry an inner ring.
      Rare super agents keep their side colour inside a yellow ring and are far
      tougher. Health bars appear once an agent is damaged. Agents only interact
      inside the active sector (amber). Faint outlines show every sector in the
      current round, and faded dots are agents last seen in previous sectors.
    </p>
  </div>
</section>
</main>
<footer>
  <p><strong>Zombie Simulation</strong></p>
  <p id="run-info">loading...</p>
  <p>Space = play/pause &middot; arrows = step &middot; generated __GENERATED__</p>
</footer>

<script>
const FRAMES = __FRAMES__;
const RUNINFO = __RUNINFO__;
const LAYOUTS = __LAYOUTS__;

// Map extent and grid come from the Python config via RUNINFO, so changing
// MAP_MIN/MAP_MAX never leaves the player out of sync (fallbacks keep old files
// readable).
const MAP_MIN = RUNINFO.map_min ?? -50;
const MAP_MAX = RUNINFO.map_max ?? 50;
const GRID_STEP = RUNINFO.grid_step ?? 100;
// Lines every GRID_STEP; numbers every other line so they stay readable.
const LABEL_STEP = GRID_STEP * 2;
const AGENT_STYLE = {
  human:  { fill: "#4d9fff", halo: "rgba(77,159,255,0.16)", radius: 3.2, maxHp: 150 },
  zombie: { fill: "#3ddc84", halo: "rgba(61,220,132,0.16)", radius: 4.0, maxHp: 200 },
};
// Agents leaving the active sector keep fading in place briefly.
const GHOST_MS = 500;
const FADE_IN_MS = 160;
// Health bars use the real spawn maxima from the run's config.
if (RUNINFO.human_max_hp) AGENT_STYLE.human.maxHp = RUNINFO.human_max_hp;
if (RUNINFO.zombie_max_hp) AGENT_STYLE.zombie.maxHp = RUNINFO.zombie_max_hp;
if (RUNINFO.human_super_max_hp) AGENT_STYLE.human.superMaxHp = RUNINFO.human_super_max_hp;
if (RUNINFO.zombie_super_max_hp) AGENT_STYLE.zombie.superMaxHp = RUNINFO.zombie_super_max_hp;
const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const W = canvas.width, H = canvas.height, PAD = 40;

const ui = {
  round: document.getElementById("s-round"),
  sector: document.getElementById("s-sector"),
  tick: document.getElementById("s-tick"),
  humans: document.getElementById("s-humans"),
  zombies: document.getElementById("s-zombies"),
  frame: document.getElementById("s-frame"),
  state: document.getElementById("s-state"),
  note: document.getElementById("s-note"),
  events: document.getElementById("events"),
  scrub: document.getElementById("scrub"),
  play: document.getElementById("play"),
  speed: document.getElementById("speed"),
  hRound: document.getElementById("h-round"),
  hSector: document.getElementById("h-sector"),
  hHumans: document.getElementById("h-humans"),
  hZombies: document.getElementById("h-zombies"),
  hState: document.getElementById("h-state"),
};

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const ghosts = new Map();
let index = 0;
let playing = !reduceMotion;
let timer = null;

document.getElementById("run-info").textContent =
  RUNINFO.sectors + " sector(s)/round \u00b7 " + RUNINFO.agents +
  " agents \u00b7 " + RUNINFO.frames + " frames" +
  (RUNINFO.parallel
    ? " \u00b7 " + RUNINFO.parallel.backend + " x" + RUNINFO.parallel.workers
      + (RUNINFO.parallel.accel && RUNINFO.parallel.accel !== "python"
        ? " \u00b7 " + RUNINFO.parallel.accel + " nearest-enemy"
        : "")
    : "") +
  (RUNINFO.parallel && RUNINFO.parallel.seed != null
    ? " \u00b7 seed " + RUNINFO.parallel.seed
    : "");

if (FRAMES.length === 0) {
  document.getElementById("run-info").textContent = "No frames captured.";
}

ui.scrub.max = Math.max(0, FRAMES.length - 1);

function sx(x) { return PAD + (x - MAP_MIN) / (MAP_MAX - MAP_MIN) * (W - 2 * PAD); }
function sy(y) { return PAD + (MAP_MAX - y) / (MAP_MAX - MAP_MIN) * (H - 2 * PAD); }

function drawGrid() {
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let v = MAP_MIN; v <= MAP_MAX; v += GRID_STEP) {
    ctx.beginPath();
    ctx.moveTo(sx(v), sy(MAP_MAX));
    ctx.lineTo(sx(v), sy(MAP_MIN));
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(sx(MAP_MIN), sy(v));
    ctx.lineTo(sx(MAP_MAX), sy(v));
    ctx.stroke();
  }

  ctx.strokeStyle = "rgba(255,255,255,0.22)";
  ctx.strokeRect(sx(MAP_MIN), sy(MAP_MAX),
                 sx(MAP_MAX) - sx(MAP_MIN), sy(MAP_MIN) - sy(MAP_MAX));

  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "11.5px ui-monospace, monospace";
  for (let v = MAP_MIN; v <= MAP_MAX; v += LABEL_STEP) {
    ctx.fillText(String(v), sx(v) - 10, H - PAD + 18);
    ctx.fillText(String(v), 6, sy(v) + 4);
  }
}

function sectorRect(bounds) {
  const [x0, y0, x1, y1] = bounds;
  return [sx(x0), sy(y1), sx(x1) - sx(x0), sy(y0) - sy(y1)];
}

// Every sector in the round stays visible, so the map does not strobe as the
// simulation moves between sectors. Only the active one is highlighted.
function drawSectors(meta) {
  const layout = LAYOUTS[meta.round] || [];
  for (const bounds of layout) {
    if (!bounds) continue;
    const [x, y, w, h] = sectorRect(bounds);
    ctx.fillStyle = "rgba(255,213,79,0.03)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(255,213,79,0.16)";
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
  }

  if (!meta.bounds) return;
  const [x, y, w, h] = sectorRect(meta.bounds);
  ctx.fillStyle = "rgba(255,213,79,0.10)";
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = "rgba(255,213,79,0.85)";
  ctx.lineWidth = 1.6;
  ctx.strokeRect(x, y, w, h);

  const label = "Sector " + meta.sector;
  ctx.font = "600 11px ui-monospace, monospace";
  const labelW = ctx.measureText(label).width + 12;
  roundRect(x + 6, y + 6, labelW, 18, 9);
  ctx.fillStyle = "rgba(11,16,32,0.82)";
  ctx.fill();
  ctx.strokeStyle = "rgba(255,213,79,0.45)";
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.fillStyle = "#ffd54f";
  ctx.fillText(label, x + 12, y + 18.5);
}

function roundRect(x, y, w, h, radius) {
  const r = Math.min(radius, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawAgent(agent, alpha) {
  const [, type, x, y, hp, , isSuper] = agent;
  const px = sx(x), py = sy(y);
  const style = type === "human" ? AGENT_STYLE.human : AGENT_STYLE.zombie;
  ctx.globalAlpha = alpha;

  // Soft halo instead of an expensive per-agent shadow/gradient.
  ctx.beginPath();
  ctx.arc(px, py, style.radius + 2, 0, Math.PI * 2);
  ctx.fillStyle = style.halo;
  ctx.fill();

  // Body.
  ctx.beginPath();
  ctx.arc(px, py, style.radius, 0, Math.PI * 2);
  ctx.fillStyle = style.fill;
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = "rgba(0,0,0,0.45)";
  ctx.stroke();

  // Shape cue as well as colour: zombies carry an inner ring.
  if (type !== "human") {
    ctx.beginPath();
    ctx.arc(px, py, style.radius - 1.4, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(9,40,24,0.45)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Specular highlight.
  ctx.beginPath();
  ctx.arc(px - style.radius * 0.32, py - style.radius * 0.38,
          style.radius * 0.32, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.fill();

  // Super agents keep their side colour inside a yellow ring.
  if (isSuper) {
    ctx.beginPath();
    ctx.arc(px, py, style.radius + 2.6, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffd54f";
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  // Health bar only once damaged: a bar over every agent is most of the noise.
  const maxHp = isSuper ? (style.superMaxHp || style.maxHp) : style.maxHp;
  const fraction = Math.max(0, Math.min(1, hp / maxHp));
  if (fraction < 1) {
    const barW = 10, barH = 2.5;
    const barX = px - barW / 2;
    const barY = py - style.radius - 5;

    roundRect(barX, barY, barW, barH, barH / 2);
    ctx.fillStyle = "rgba(255,255,255,0.18)";
    ctx.fill();

    if (fraction > 0) {
      roundRect(barX, barY, barW * fraction, barH, barH / 2);
      ctx.fillStyle = fraction > 0.5 ? "#5ddb8a"
                    : fraction > 0.25 ? "#ffc857" : "#ff6b6b";
      ctx.fill();
    }
  }

  ctx.globalAlpha = 1;
}

function drawAgents(agents) {
  const live = new Set();
  for (const agent of agents) live.add(agent[0]);
  const now = performance.now();

  // Agents from the sector that was just simulated fade out in place.
  for (const [id, ghost] of ghosts) {
    if (live.has(id)) continue;
    const age = now - ghost.seenAt;
    if (age < 0 || age > GHOST_MS) {
      ghosts.delete(id);
      continue;
    }
    const style = ghost.type === "human" ? AGENT_STYLE.human : AGENT_STYLE.zombie;
    ctx.globalAlpha = 0.45 * (1 - age / GHOST_MS);
    ctx.beginPath();
    ctx.arc(sx(ghost.x), sy(ghost.y), style.radius * 0.9, 0, Math.PI * 2);
    ctx.fillStyle = style.fill;
    ctx.fill();
  }
  ctx.globalAlpha = 1;

  for (const agent of agents) {
    const [id, type, x, y] = agent;
    let record = ghosts.get(id);
    if (!record) {
      record = { type, x, y, seenAt: now, bornAt: now };
      ghosts.set(id, record);
    } else if (now - record.seenAt > GHOST_MS) {
      record.bornAt = now;  // returning after a fade-out fades back in
    }
    record.type = type;
    record.x = x;
    record.y = y;
    record.seenAt = now;
    const fadeIn = Math.min(1, 0.15 + (now - record.bornAt) / FADE_IN_MS);
    drawAgent(agent, fadeIn);
  }
}

function render() {
  drawGrid();
  const frame = FRAMES[index];
  if (!frame) return;

  drawSectors(frame.meta);
  drawAgents(frame.agents);
  updatePanel(frame.meta);
}

function updatePanel(meta) {
  const round = meta.round || 1;
  const sectors = meta.sectors || RUNINFO.sectors;
  const state = meta.combat ? "combat" : "quiet";

  ui.round.textContent = round;
  ui.sector.textContent = meta.sector + "/" + sectors;
  ui.tick.textContent = meta.iteration;
  ui.humans.textContent = meta.humans_alive;
  ui.zombies.textContent = meta.zombies_alive;
  ui.frame.textContent = (index + 1) + "/" + FRAMES.length;

  ui.state.textContent = state;
  ui.state.className = "pill " + state;
  ui.note.textContent = meta.note || "";

  ui.hRound.textContent = "Round " + round;
  ui.hSector.textContent = "Sector " + meta.sector + "/" + sectors;
  ui.hHumans.textContent = meta.humans_alive;
  ui.hZombies.textContent = meta.zombies_alive;
  ui.hState.textContent = state;
  ui.hState.className = "pill " + state;

  canvas.setAttribute(
    "aria-label",
    "Round " + round + ", sector " + meta.sector + " of " + sectors + ", " +
    meta.humans_alive + " humans and " + meta.zombies_alive +
    " zombies alive, " + state + "."
  );

  const events = meta.events || [];
  if (events.length === 0) {
    ui.events.innerHTML = '<li class="muted">No combat this tick.</li>';
  } else {
    ui.events.innerHTML = "";
    for (const event of events) {
      const li = document.createElement("li");
      li.textContent = event;
      ui.events.appendChild(li);
    }
  }
}

function setFrame(next) {
  const clamped = Math.max(0, Math.min(FRAMES.length - 1, next));
  // Jumping or stepping backwards must not leave stale ghost trails.
  if (clamped !== index + 1) ghosts.clear();
  index = clamped;
  ui.scrub.value = index;
  render();
}

function setPlaying(value) {
  playing = value;
  ui.play.textContent = playing ? "Pause" : "Play";
  schedule();
}

function schedule() {
  clearTimeout(timer);
  if (!playing || FRAMES.length === 0) return;
  const speed = parseFloat(ui.speed.value) || 1;
  timer = setTimeout(() => {
    if (index >= FRAMES.length - 1) {
      setPlaying(false);
      return;
    }
    setFrame(index + 1);
    schedule();
  }, Math.max(20, 240 / speed));
}

ui.play.addEventListener("click", () => {
  if (!playing && index >= FRAMES.length - 1) setFrame(0);
  setPlaying(!playing);
});
document.getElementById("prev").addEventListener("click", () => { setPlaying(false); setFrame(index - 1); });
document.getElementById("next").addEventListener("click", () => { setPlaying(false); setFrame(index + 1); });
document.getElementById("restart").addEventListener("click", () => { setFrame(0); setPlaying(true); });
ui.scrub.addEventListener("input", () => { setPlaying(false); setFrame(parseInt(ui.scrub.value, 10)); });
ui.speed.addEventListener("change", () => schedule());

window.addEventListener("keydown", (event) => {
  const tag = (event.target && event.target.tagName) || "";
  if (tag === "INPUT" || tag === "SELECT") return;
  // A focused button handles its own Space activation.
  if (tag === "BUTTON" && event.code === "Space") return;

  if (event.code === "Space") {
    event.preventDefault();
    if (!playing && index >= FRAMES.length - 1) setFrame(0);
    setPlaying(!playing);
  } else if (event.code === "ArrowRight") {
    setPlaying(false);
    setFrame(index + 1);
  } else if (event.code === "ArrowLeft") {
    setPlaying(false);
    setFrame(index - 1);
  }
});

ui.play.textContent = playing ? "Pause" : "Play";
setFrame(0);
schedule();
</script>
</body>
</html>
"""


def _grid_step(span: int, divisions: int = 20) -> int:
    """Round grid step giving roughly ``divisions`` cells across the map."""
    if span <= 0 or divisions <= 0:
        return 1
    raw = max(1, span // divisions)
    magnitude = 10 ** (len(str(raw)) - 1)
    return max(magnitude, round(raw / magnitude) * magnitude)


def _clear_old_runs(directory: Path, keep: Path, since: float) -> None:
    """Delete players from before this run started, leaving ``keep``.

    A player written by another process *while* this run was going is not
    "old": deleting it would silently remove that run's output.
    """
    for old in directory.glob("simulation_*.html"):
        if old == keep:
            continue
        try:
            if old.stat().st_mtime >= since:
                continue
            old.unlink()
        except OSError:
            # A browser still holding the file must not fail the run.
            pass


class HtmlVisualizer:
    """Collects frames and writes a standalone HTML player at the end."""

    def __init__(self, path: str | Path | None = None):
        # Only the default output folder is pruned; an explicit path may point
        # at a directory this run does not own.
        self._output_dir: Path | None = None
        self._started = time.time()
        if path is None:
            # Project-relative so the file does not land in whatever directory
            # the app was started from; microseconds keep runs from colliding.
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            root = Path(__file__).resolve().parents[1]
            self._output_dir = root / cfg.HTML_OUTPUT_DIR
            path = self._output_dir / f"simulation_{stamp}.html"
        self.path = Path(path)
        self.frames: list[dict] = []
        # Sector rectangles per round, collected from frame metadata so every
        # sector stays visible even when frames are sampled away.
        self.layouts: dict[int, list[list[float]]] = {}
        self.sector_count = 0
        self.agent_count = 0
        self.info: dict = {}
        self._written = False
        self._records = 0

    # ------------------------------------------------------------------ #
    def start(self, sector_count: int, agent_count: int, info: dict | None = None) -> None:
        self.sector_count = sector_count
        self.agent_count = agent_count
        self.info = info or {}

    def frame(self, snapshot: list[tuple], meta: dict) -> None:
        round_index = int(meta.get("round", 1) or 1)
        bounds = meta.get("bounds")
        if bounds:
            layout = self.layouts.setdefault(round_index, [])
            if bounds not in layout:
                layout.append(bounds)
        frame = {"meta": meta, "agents": [list(agent) for agent in snapshot]}
        self._records += len(frame["agents"])
        # Keep a sample of the whole run: halve the stored frames when a budget
        # is hit, instead of dropping everything after the cap.
        if len(self.frames) > 1 and (
            len(self.frames) >= cfg.MAX_FRAMES
            or self._records > cfg.MAX_FRAME_RECORDS
        ):
            self.frames = self.frames[::2]
            self._records = sum(len(item["agents"]) for item in self.frames)
        if len(self.frames) >= cfg.MAX_FRAMES:
            if meta.get("ended") and self.frames:
                self.frames[-1] = frame
            return
        self.frames.append(frame)

    def finish(self, prune: bool = True) -> None:
        if self._written:
            return
        self._written = True
        # A one-sided run never starts the visualiser, so it must not replace
        # a real player with an empty page.
        if not self.frames:
            return

        min_x, min_y, max_x, max_y = self.info.get("map_bounds") or map_bounds_for(0)
        run_info = {
            "sectors": self.sector_count,
            "agents": self.agent_count,
            "frames": len(self.frames),
            "map_min": min_x,
            "map_max": max_x,
            "grid_step": _grid_step(int(max_x - min_x)),
            "human_max_hp": cfg.HUMAN_HEALTH_RANGE[1],
            "zombie_max_hp": cfg.ZOMBIE_HEALTH_RANGE[1],
            "human_super_max_hp": (
                cfg.HUMAN_HEALTH_RANGE[1] * cfg.SUPER_HEALTH_MULTIPLIER
            ),
            "zombie_super_max_hp": (
                cfg.ZOMBIE_HEALTH_RANGE[1] * cfg.SUPER_HEALTH_MULTIPLIER
            ),
            "parallel": self.info,
        }
        html = (
            HTML_TEMPLATE.replace("__FRAMES__", json.dumps(self.frames, separators=(",", ":")))
            .replace("__RUNINFO__", json.dumps(run_info))
            .replace("__LAYOUTS__", json.dumps(self.layouts, separators=(",", ":")))
            .replace("__TITLE__", "Zombie Simulation")
            .replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(html, encoding="utf-8")
        # Write first, then prune: a failed write must not leave zero players.
        # An aborted run passes prune=False so the last complete player stays.
        if prune and self._output_dir is not None:
            _clear_old_runs(self._output_dir, self.path, self._started)
