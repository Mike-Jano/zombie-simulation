"""Self-contained HTML animation of a completed simulation run."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config import simulation_config as cfg

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
    --muted: #8b93ad;
    --human: #4d9fff;
    --human-soft: rgba(77,159,255,.18);
    --zombie: #3ddc84;
    --zombie-soft: rgba(61,220,132,.18);
    --accent: #ffd54f;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font-family: "Segoe UI", system-ui, -apple-system, sans-serif; }
  header { padding: 20px 24px 4px; }
  header h1 { margin: 0 0 6px; font-size: 20px; letter-spacing: .04em; }
  header p { margin: 0; color: var(--muted); font-size: 13px; }
  main { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 20px;
         padding: 16px 24px 8px; align-items: start; }
  #stage { background: #070b16; border: 1px solid var(--line); border-radius: 12px; padding: 8px; }
  canvas { display: block; width: 100%; height: auto; border-radius: 8px; }
  aside { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  @media (max-width: 960px) { main { grid-template-columns: 1fr; } }
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
  .pill.combat { background: rgba(255,107,107,.14); color: #ff9d9d; border-color: rgba(255,107,107,.4); }
  .pill.quiet { background: rgba(77,159,255,.12); color: #8cc4ff; border-color: rgba(77,159,255,.35); }
  #events { list-style: none; margin: 0; padding: 0; font-size: 12px; line-height: 1.5;
            color: var(--muted); max-height: 200px; overflow: auto; }
  #events li { border-bottom: 1px dashed rgba(255,255,255,.06); padding: 3px 0; }
  #events li:last-child { border-bottom: none; }
  #controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 8px 24px 28px; }
  button { background: #1d2742; color: var(--text); border: 1px solid var(--line);
           border-radius: 8px; padding: 8px 14px; font-size: 13px; cursor: pointer; }
  button:hover { background: #253155; }
  button.primary { background: var(--accent); color: #1a1a1a; border-color: var(--accent); font-weight: 600; }
  select { background: #1d2742; color: var(--text); border: 1px solid var(--line);
           border-radius: 8px; padding: 8px; font-size: 13px; }
  #scrub { flex: 1; min-width: 220px; accent-color: var(--accent); }
  .legend { display: flex; gap: 16px; font-size: 12px; color: var(--muted);
            align-items: center; flex-wrap: wrap; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .dot.human { background: var(--human); box-shadow: 0 0 0 3px var(--human-soft); }
  .dot.zombie { background: var(--zombie); box-shadow: 0 0 0 3px var(--zombie-soft); }
  .dot.dead { background: transparent; border: 1.5px solid rgba(170,178,200,.45); }
  footer { padding: 0 24px 24px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1>Zombie Simulation</h1>
  <p id="run-info">loading...</p>
</header>
<main>
  <div id="stage"><canvas id="map" width="900" height="900"></canvas></div>
  <aside>
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
      <h2>Legend</h2>
      <div class="legend">
        <span><span class="dot human"></span>Human</span>
        <span><span class="dot zombie"></span>Zombie</span>
        <span><span class="dot dead"></span>Dead</span>
        <span><span class="dot" style="background:var(--accent);border-radius:2px"></span>Sector</span>
      </div>
      <p class="muted" style="font-size:12px; margin:12px 0 0; line-height:1.5;">
        Blue dots are humans, green dots are zombies. The bar above each dot is
        its health; agents only ever interact with others in the same sector.
      </p>
    </div>
  </aside>
</main>
<div id="controls">
  <button id="play" class="primary">Pause</button>
  <button id="prev">Prev</button>
  <button id="next">Next</button>
  <button id="restart">Restart</button>
  <input id="scrub" type="range" min="0" max="0" value="0" step="1">
  <select id="speed">
    <option value="0.5">0.5x</option>
    <option value="1">1x</option>
    <option value="2">2x</option>
    <option value="4" selected>4x</option>
    <option value="8">8x</option>
    <option value="16">16x</option>
  </select>
</div>
<footer>Space = play/pause &middot; arrows = step &middot; generated __GENERATED__</footer>

<script>
const FRAMES = __FRAMES__;
const RUNINFO = __RUNINFO__;

// Map extent and grid come from the Python config via RUNINFO, so changing
// MAP_MIN/MAP_MAX never leaves the player out of sync (fallbacks keep old files
// readable).
const MAP_MIN = RUNINFO.map_min ?? -500;
const MAP_MAX = RUNINFO.map_max ?? 500;
const GRID_STEP = RUNINFO.grid_step ?? 100;
// Lines every GRID_STEP; numbers every other line so they stay readable.
const LABEL_STEP = GRID_STEP * 2;
const AGENT_STYLE = {
  human:  { fill: "#4d9fff", halo: "rgba(77,159,255,0.18)", radius: 5.5, maxHp: 150 },
  zombie: { fill: "#3ddc84", halo: "rgba(61,220,132,0.18)", radius: 6.5, maxHp: 200 },
};
// Health bars use the real spawn maxima from the run's config.
if (RUNINFO.human_max_hp) AGENT_STYLE.human.maxHp = RUNINFO.human_max_hp;
if (RUNINFO.zombie_max_hp) AGENT_STYLE.zombie.maxHp = RUNINFO.zombie_max_hp;
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
};

let index = 0;
let playing = true;
let timer = null;

document.getElementById("run-info").textContent =
  RUNINFO.sectors + " sector(s)/round \u00b7 " + RUNINFO.agents +
  " agents \u00b7 " + RUNINFO.frames + " frames" +
  (RUNINFO.parallel
    ? " \u00b7 " + RUNINFO.parallel.backend + " x" + RUNINFO.parallel.workers
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

  ctx.fillStyle = "rgba(255,255,255,0.35)";
  ctx.font = "10.5px ui-monospace, monospace";
  for (let v = MAP_MIN; v <= MAP_MAX; v += LABEL_STEP) {
    ctx.fillText(String(v), sx(v) - 10, H - PAD + 18);
    ctx.fillText(String(v), 6, sy(v) + 4);
  }
}

function drawSector(bounds) {
  if (!bounds) return;
  const [x0, y0, x1, y1] = bounds;
  ctx.save();
  ctx.setLineDash([7, 7]);
  ctx.strokeStyle = "rgba(255,213,79,0.7)";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(sx(x0), sy(y1), sx(x1) - sx(x0), sy(y0) - sy(y1));
  ctx.restore();
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

function drawAgents(agents) {
  for (const agent of agents) {
    const [, type, x, y, hp, alive] = agent;
    const px = sx(x), py = sy(y);
    const style = type === "human" ? AGENT_STYLE.human : AGENT_STYLE.zombie;

    if (!alive) {
      // Hollow ring: clearly present, clearly out of the fight.
      ctx.beginPath();
      ctx.arc(px, py, 3.2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(170,178,200,0.35)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      continue;
    }

    // Soft halo instead of an expensive per-agent shadow/gradient.
    ctx.beginPath();
    ctx.arc(px, py, style.radius + 4.5, 0, Math.PI * 2);
    ctx.fillStyle = style.halo;
    ctx.fill();

    // Body.
    ctx.beginPath();
    ctx.arc(px, py, style.radius, 0, Math.PI * 2);
    ctx.fillStyle = style.fill;
    ctx.fill();
    ctx.lineWidth = 1.4;
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.stroke();

    // Shape cue as well as colour: zombies carry an inner ring.
    if (type !== "human") {
      ctx.beginPath();
      ctx.arc(px, py, style.radius - 2.2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(9,40,24,0.45)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // Specular highlight.
    ctx.beginPath();
    ctx.arc(px - style.radius * 0.32, py - style.radius * 0.38,
            style.radius * 0.34, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.fill();

    // Health bar: rounded, capped at that type's spawn maximum.
    const fraction = Math.max(0, Math.min(1, hp / style.maxHp));
    const barW = 16, barH = 3.5;
    const barX = px - barW / 2;
    const barY = py - style.radius - 7;

    roundRect(barX, barY, barW, barH, barH / 2);
    ctx.fillStyle = "rgba(255,255,255,0.16)";
    ctx.fill();

    if (fraction > 0) {
      roundRect(barX, barY, barW * fraction, barH, barH / 2);
      ctx.fillStyle = fraction > 0.5 ? "#5ddb8a"
                    : fraction > 0.25 ? "#ffc857" : "#ff6b6b";
      ctx.fill();
    }
  }
}

function render() {
  drawGrid();
  const frame = FRAMES[index];
  if (!frame) return;

  drawSector(frame.meta.bounds);
  drawAgents(frame.agents);
  updatePanel(frame.meta);
}

function updatePanel(meta) {
  ui.round.textContent = meta.round || 1;
  ui.sector.textContent =
    meta.sector + "/" + (meta.sectors || RUNINFO.sectors);
  ui.tick.textContent = meta.iteration;
  ui.humans.textContent = meta.humans_alive;
  ui.zombies.textContent = meta.zombies_alive;
  ui.frame.textContent = (index + 1) + "/" + FRAMES.length;

  ui.state.textContent = meta.combat ? "combat" : "quiet";
  ui.state.className = "pill " + (meta.combat ? "combat" : "quiet");
  ui.note.textContent = meta.note || "";

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
  index = Math.max(0, Math.min(FRAMES.length - 1, next));
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


class HtmlVisualizer:
    """Collects frames and writes a standalone HTML player at the end."""

    def __init__(self, path: str | Path | None = None):
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = Path(cfg.HTML_OUTPUT_DIR) / f"simulation_{stamp}.html"
        self.path = Path(path)
        self.frames: list[dict] = []
        self.sector_count = 0
        self.agent_count = 0
        self.info: dict = {}
        self._written = False

    # ------------------------------------------------------------------ #
    def start(self, sector_count: int, agent_count: int, info: dict | None = None) -> None:
        self.sector_count = sector_count
        self.agent_count = agent_count
        self.info = info or {}

    def frame(self, snapshot: list[tuple], meta: dict) -> None:
        if len(self.frames) >= cfg.MAX_FRAMES:
            return
        self.frames.append(
            {"meta": meta, "agents": [list(agent) for agent in snapshot]}
        )

    def finish(self) -> None:
        if self._written:
            return
        self._written = True

        run_info = {
            "sectors": self.sector_count,
            "agents": self.agent_count,
            "frames": len(self.frames),
            "map_min": cfg.MAP_MIN,
            "map_max": cfg.MAP_MAX,
            "grid_step": _grid_step(cfg.MAP_SPAN),
            "human_max_hp": cfg.HUMAN_HEALTH_RANGE[1],
            "zombie_max_hp": cfg.ZOMBIE_HEALTH_RANGE[1],
            "parallel": self.info,
        }
        html = (
            HTML_TEMPLATE.replace("__FRAMES__", json.dumps(self.frames, separators=(",", ":")))
            .replace("__RUNINFO__", json.dumps(run_info))
            .replace("__TITLE__", "Zombie Simulation")
            .replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(html, encoding="utf-8")
