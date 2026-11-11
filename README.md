# Zombie Simulation (Python + PostgreSQL)

A 2D agent simulation where human and zombie agents live in PostgreSQL, are
loaded into Python a sector at a time, fight until one side is wiped out, and
are written back to the database before being unloaded from memory.

---


### Prerequisites

- Python 3.9+ (the bundled venv is 3.14)
- A running PostgreSQL server
- `psycopg` 3 — **required**. It is imported at start-up, so the program cannot
  run without it; there is no pure-Python fallback (unlike `numpy`/`wgpu`).
  Install it with a copy-paste of this one line in your activated venv:

  ```
  pip install "psycopg[binary]"
  ```

  The `[binary]` extra pulls a prebuilt wheel, so no compiler or local
  PostgreSQL headers are needed. To install the driver plus the optional
  accelerators in one go:

  ```
  pip install -r requirements.txt
  ```
- Optional: `numpy` (fast CPU nearest-enemy search) and `wgpu` (WebGPU compute
  on AMD/Intel/NVIDIA/Apple GPUs). Both are in `requirements.txt`; without them
  the simulation runs on the pure-Python scalar path.

You do **not** need to create the database by hand. On start-up the app creates
the `zombie_sim` database and the `agents` / `humans` / `zombies` tables if they
are missing. Only the connection credentials are needed.

---

## Database login

All login details live in **one hand-editable file**:

```
config\database.ini
```

Open it and fill in your PostgreSQL details:

```ini
[postgresql]
host = localhost
port = 5432
dbname = zombie_sim
user = postgres
password = postgres
```

That is the whole configuration. A few notes:

- **`dbname` is created automatically** if it does not exist, so it does not
  have to be there yet — your user just needs permission to create databases.
- **Leave `password` blank** to use trust authentication or a `.pgpass` file.
- **Menu option 3 ("Database settings")** edits this same file from inside the
  app, with a hidden password prompt, then verifies the connection.
- **The app prints the full path** to this file at the top of every run, and in
  the menu header, so it is always obvious what is being used.

Values left blank in the file fall back to the environment variables `DB_HOST`,
`DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, and then to built-in defaults.
The file always wins, so editing it is enough.

> The password is stored in plain text. This file is specific to your machine —
> don't commit it to a public repository.

Connection attempts time out after 5 seconds (`CONNECT_TIMEOUT` in
`config/db_config.py`), so a wrong host or port reports an error instead of
hanging.

---

## Using it

Starting `main.py` shows a menu:

```
Database: postgres@localhost:5432/zombie_sim
  1) Start new simulation
  2) Clear & recreate database
  3) Database settings
  4) Database status
  5) Exit (clears agents)
```

- **1) Start new simulation** – prompts for the number of humans and zombies,
  the visualisation mode, and an optional random seed (same seed = same run).
- **2) Clear & recreate database** – drops and rebuilds the three tables at will.
- **3) Database settings** – host / port / database name / user / password. Writes
  `config/database.ini` (the password is entered with hidden input) and tests the
  connection.
- **4) Database status** – alive/dead counts per type.
- **5) Exit** – clears all agents before quitting, as required. `Ctrl+C` does the
  same thing.

### Visualisation modes

| Choice | What you get |
| --- | --- |
| 1 | A self-contained HTML file in `output/`, opened in your browser |
| 2 | Headless – sector summaries and the final report only |

The HTML player has play/pause, prev/next, restart, a scrub slider, a speed
selector (0.5x–16x), and keyboard controls (Space, arrow keys). Humans are blue
and zombies green; a health bar appears
over a dot once it is damaged. Every sector in the current round is outlined,
the active one is highlighted in amber, and faded dots are agents last seen in
previous sectors. It embeds sampled frames (every `FRAME_INTERVAL` ticks, capped
at `MAX_FRAMES`, always keeping the first and last), so it works offline and can
be moved or shared as a single file. Each new run replaces the older generated
players in `output/`.

---

## How the simulation works

### Data model

Single-table inheritance, so a subtable row can never disagree with the
supertable about an agent's type:

```
agents   (agent_id PK, agent_type, is_alive, x_pos, y_pos, health, speed)
humans   (agent_id, agent_type='human',  weapon_tier)  -> FK to agents
zombies  (agent_id, agent_type='zombie', strength)     -> FK to agents
```

The subtables reference `agents` through the composite key
`(agent_id, agent_type)` with `ON DELETE CASCADE`.

Dead agents are **never deleted** – `is_alive` is just set to `false`, so the
database keeps the full history of the run. They are not loaded back into
memory and do not appear in the HTML frames; only living agents are read,
simulated and drawn.

Only one simulation can use a database at a time. The app takes a PostgreSQL
advisory lock for the run and refuses a second run (or a reset) instead of
truncating the first run's agents.

### Memory lifecycle

Agents are not all loaded at once. `src/simulation.py` works in **rounds**:

1. Read every living agent's position (`SELECT ... WHERE is_alive`).
2. Cut those positions into sectors with a grid (`utils/sectoring.py`). The
   sector count scales with the population — one sector per
   `AGENTS_PER_SECTOR` agents, so 200 agents are a single arena (see
   [Parallelism](#parallelism)).
3. For each sector: load only its agents into `Human` / `Zombie` objects,
   simulate at most `SECTOR_REFRESH_TICKS` ticks, write positions, health and
   `is_alive` back to PostgreSQL, then drop the objects from memory.
4. After the last sector, **re-cluster from the agents' current positions** and
   begin the next round.

Sectors are therefore not fixed for the whole run — they are rebuilt every
`SECTOR_REFRESH_TICKS` ticks, so they keep following the action instead of
drifting away from where the agents actually are.

A sector's walkable area is its cluster's bounding box plus padding, clamped to
the map. Agents cannot leave the map or their sector for the duration of a
visit. The map is sized once from the starting population (see
[Map size](#map-size)).

### Tick order

Each tick runs movement, then combat, simultaneously for both sides:

- **Movement** – every living agent walks toward the nearest enemy in its
  sector, at most `speed` units. Humans stop as soon as the target is inside
  their weapon range and fire from there; zombies keep closing to 1 unit until
  they are in biting range.
- **Combat** – all attacks are resolved against the positions at the start of
  the tick, then damage is applied together. A zombie killed this tick still
  lands its attack.

Internally each phase is split into **plan** and **apply**: planning reads the
snapshot and returns per-agent proposals (a move delta or an attack target),
and only the apply step mutates agents, in ascending `agent_id` order. That
split is what makes the tick safe to shard without changing behaviour — see
[Tick-level sharding](#tick-level-sharding-inside-a-sector).

### Combat rules

| | Humans | Zombies |
| --- | --- | --- |
| Range | `weapon_tier * 4` units (4–20) | fixed 2.5 units |
| Damage | random, `tier*3 … tier*9` | random, `strength*5 … strength*15` |
| Accuracy | roll to hit; falls off with distance, higher tier keeps accuracy further out | always hits in range |

A tier-5 human averages 30 damage per hit, so a zombie (50–200 HP) takes
several shots. A zombie's bite averages 10 damage per strength point (30 at
strength 3, 50 at strength 5), so a human that lets it close usually dies in
about five bites. Humans still out-range zombies — they get free shots while
the horde walks in — but they can no longer mow it down before it arrives.

Rare **super agents** (default `0.00002` per spawned agent, either side) get
double the spawn-max tier/strength (10) and triple the spawn-max health (450
for a human, 600 for a zombie), with speed fixed at 3. They use the same combat
formulas, so a super human hits for 30–90 at 40 units. The DB has no super
column: a stat above the normal spawn range is the marker, and the HTML player
draws a yellow ring around them.

### End of a sector visit

Whichever happens first:

- one side is eliminated,
- no combat for `NO_COMBAT_PATIENCE` consecutive ticks after fighting started,
- nothing moved and nothing fought,
- the closest human/zombie pair has not got any closer for `STALL_PATIENCE`
  ticks (this is what stops a sector that never engages),
- `SECTOR_REFRESH_TICKS` is reached — the normal case, because sectors are
  refreshed rather than run to exhaustion.

### Why the run cannot freeze

Every visit is capped at `min(SECTOR_REFRESH_TICKS,
MAX_ITERATIONS_PER_SECTOR)` (15 ticks by default), so no single sector can
grind on. On top of that the whole run is bounded by `MAX_ROUNDS`
(400) and `MAX_TOTAL_TICKS` (derived as
`MAX_ROUNDS * max(1, min(SECTOR_REFRESH_TICKS, MAX_ITERATIONS_PER_SECTOR))`,
6000 by default). A round in which
nothing moved and nothing fought does **not** end the run any more: the next
round collapses to a single
global sector, where the two sides are guaranteed to engage (`COLLAPSE_ON_
STAGNATION`). Only a second stalled round after that stops the run.

Before this change a sector that never engaged could only exit through
`MAX_ITERATIONS_PER_SECTOR` (2000), emitting a frame every tick and producing
enormous HTML files.

---

## Parallelism

Sectors are independent: the grid gives every agent exactly one sector, and a
visit loads, simulates and writes its agents back before the next one starts.
Nothing in a round touches another sector's agents, so a round's sectors are
already safe to run at the same time — that is the entire speed-up.

### Which backend runs

Python threads cannot run interpreted code in parallel on a normal build (the
GIL), so the backend is chosen per machine at start-up:

| Machine | Backend |
| --- | --- |
| CPython with the GIL (the default everywhere) | **processes**, started with `spawn` |
| free-threaded CPython (`python3.14t`, `sys._is_gil_enabled()` is false) | **threads** |
| one usable CPU, one sector, `--workers 1`, or a pool that fails to start | **serial** |

`os.process_cpu_count()` is used instead of `os.cpu_count()`, so containers and
hybrid big.LITTLE chips only count the cores the process may actually use.
Nothing here is architecture-specific: no `fork` (unsafe with live psycopg
connections, and absent on Windows), no CPU affinity calls, no SIMD and no
third-party packages.

### Tick-level sharding (inside a sector)

Sector-level parallelism runs out of work when the population produces only one
or two sectors. The engine therefore also shards **one sector's tick** across
workers:

- movement planning (humans against the pre-move zombie snapshot, then zombies
  against the post-move human snapshot), and
- combat targeting (every attacker against the same snapshot),

are split into contiguous chunks of agents and executed by a `thread`
executor. RNG draws and damage application stay in the serial merge step, so
chunking can never change a seeded run.

The two levels share **one** worker budget instead of nesting pools:

```
outer_workers = min(workers, sectors)
event_shards  = workers // outer_workers
```

So 12 workers and 3 sectors means 3 outer threads × 4 shards per sector, while
a single sector gets all 12 shards and no outer pool. Process and serial
backends always use one shard, because a process already owns a whole sector.
`--event-shards N` is capped by each sector's share of the worker budget, so a
typo cannot create thousands of threads. If a stalled round collapses the run to
one sector, the outer pool is closed and that sector gets the whole thread
budget.

Why shards only help on a free-threaded build: on a GIL build the shards run
correctly but one at a time, so the default (`EVENT_PARALLEL = "auto"`) leaves
them at 1 unless the resolved backend is `thread`. Pass
`--backend thread --event-shards N` to try them (or to run DB-bound work).

Shard counts are irrelevant to the result: 1, 2, 3, 4, 8 and 16 shards have
been verified byte-identical, including frames and database rows.
`EVENT_CHUNK_SIZE` adds a load-balancing cap (agents per chunk) that only ever
*raises* the chunk count.

### Finding the nearest enemy

Every agent asks for its closest living enemy twice per tick (movement and
combat). That used to be an O(n²) scan per sector; it is now answered by a
uniform-grid `SpatialIndex` (`utils/spatial.py`) in roughly O(1) expected time.
The index breaks ties on `(distance², agent_id)`, exactly like the old loop, so
the chosen target is unchanged — `tests/test_spatial.py` checks it against a
brute-force reference on random, collinear and duplicate layouts.

That query is also the only part of a tick that can be answered for a whole
group at once, so an optional accelerator replaces the per-agent loop with one
batched call (`runtime/accelerator.py`, `utils/spatial_batch.py`):

| `--accel` | Backend | Composes with the worker pool? |
| --- | --- | --- |
| `off` | reference scalar grid | yes |
| `auto` (default) | numpy when installed, else scalar | yes |
| `numpy` | numpy float64, tiled argmin | yes |
| `gpu` | WebGPU compute shader via `wgpu-py` | no: one device, so the run goes serial |

Both `numpy` and `wgpu` are optional at runtime; a missing module, missing
adapter, software rasteriser, shader error or lost device all fall back to the
scalar path and log once. `auto` never touches the GPU stack: enumerating
adapters costs about a second on first use, and numpy composes with the process
pool while a GPU run must be single-threaded. Ask for the GPU with
`--accel gpu`.

**The GPU cannot change a result.** WebGPU computes in f32, so it only nominates
a winner; the choice is confirmed on the CPU in float64, and any query whose
runner-up is inside the f32 error band is recomputed exactly. When an adapter is
present, a `gpu` request also calibrates the device against numpy once (250
queries × 250 targets) and demotes to numpy if the GPU loses. `python`, `numpy`
and `gpu` runs therefore produce the same frames, counts and final states that
get saved — `tests/test_accel.py` proves it at the sector-visit level. The report
line and the HTML footer still name the backend that ran, so those lines differ
by design.

Only the target *lookup* is batched. Movement arithmetic, RNG draws and damage
application stay in their original order, which is why the backend can be
switched without invalidating a seed.

Measured on the reference laptop (AMD Radeon 840M iGPU, 12 logical CPUs), with
auto sectoring keeping groups at ~100–300 agents per side, numpy is the clear
winner and the GPU loses its dispatch overhead:

| Starting agents | scalar | numpy | gpu |
| --- | --- | --- | --- |
| 1200 (3 sectors) | 82 sector-ticks/s | 243 | 40 |
| 3000 (6 sectors) | 58 | 299 | 135 |
| 12000 (24 sectors) | 54 | 241 | 161 |

The GPU pays off only when a single arena is genuinely large, because one
dispatch plus its blocking readback costs ~0.6–2 ms whatever the group size.
Pinning `--sectors 1` so each side holds hundreds to thousands of agents:

| Agents in one sector | numpy | gpu |
| --- | --- | --- |
| 1200 (600 per side) | 44 sector-ticks/s | 9 |
| 3000 (1500 per side) | 10 | 22 |
| 12000 (6000 per side) | 0.8 | 8.2 |

A discrete GPU changes the constants, not the shape: the accelerator is a tool
for big sectors, and numpy remains the default because sectors are small.

### Sectors scale with the population

```
max_sectors = max(1, ceil(starting_agents / AGENTS_PER_SECTOR))
k           = clamp(ceil(living_agents / AGENTS_PER_SECTOR), 1, max_sectors)
```

There is no fixed maximum. The ceiling is derived from the starting population;
the actual count falls as agents die.

| Starting population | Maximum sectors |
| --- | --- |
| 200 | 1 |
| 500 | 1 |
| 1000 | 2 |
| 1500 | 3 |
| 3000 | 6 |
| 6000 | 12 |
| 60,000 | 120 |

For example, a run that starts with 3000 agents may use up to 6 sectors; once
only 1500 are alive it uses 3, and once only 100 are alive it uses 1.

`MAX_WORKERS` still caps parallel width, so sectors beyond the worker count
simply queue. The worker count never influences the plan: cores execute the
sectors the population produced, they do not create more. Two machines with
different CPU counts therefore simulate the same run from the same seed.

Coarse sectors are a correctness feature, not just a speed knob. A finely split
map leaves clusters containing only humans or only zombies, and such a sector
can never fight — that is how runs used to end with both sides still alive.
Every sector is also guaranteed to hold both types (single-type clusters are
merged into the nearest mixed one), and if a whole round still manages no
attack and no movement, the run **collapses to one global sector** instead of
stopping. With both types in one arena every tick either lands a hit or closes
a distance, so a run ends with one side eliminated rather than stalling.

### Reproducing a run

The seed is enough: sectoring depends only on the population, never on the
worker count, so this pair produces byte-identical database state:

```
python main.py --run --humans 1500 --zombies 1500 --seed 7 --workers 1
python main.py --run --humans 1500 --zombies 1500 --seed 7 --workers 12
```

`--sectors N` still pins the count when you want to force a particular split.
The report and the HTML footer record the backend, worker count, event-shard
count, sector count, accelerator and seed of the run. None of `--workers`,
`--backend`, `--event-shards` or `--accel` changes the result: sharding only
reorders *who computes* a proposal, never the proposal itself, and the
accelerator only changes who answers "who is closest".

### Measuring

`tools/bench_parallel.py` runs the real engine on synthetic agents (no
database) and prints the speed-up curve, so each chip can pick its own worker
count:

```
python tools/bench_parallel.py --population 1200
python tools/bench_parallel.py --population 4000 --workers 1,2,4,8,12 --rounds 4
python tools/bench_parallel.py --population 1200 --backend thread --event-shards 4
python tools/bench_parallel.py --population 1200 --accel numpy
python tools/bench_parallel.py --population 1200 --accel off
python tools/bench_parallel.py --population 3000 --accel gpu
```

`--accel` (the same names as `main.py`) measures every row with that backend,
so the same population can be compared across `off`, `numpy` and `gpu`. A GPU
run prints no pool rows because a real GPU run is serial by design; when the
calibration demotes the device, the table says `numpy (CPU)` and the pool rows
come back.

On the reference machine (12 logical CPUs, Windows): 1200 agents make 3 sectors
and scale 1x → 2.6x on 3 workers. 4000 agents make 8 sectors and reach 5.0x on
8 workers. With `--accel numpy` the same 1200-agent workload goes 238 → 540 →
791 sector-ticks/s on 1, 2 and 4 workers (2.3x and 3.3x), and the serial
baseline itself is ~3x the scalar loop's 82 sector-ticks/s. The outer pool is
capped at the sector count; on the thread backend the rest of the `--workers`
budget becomes per-sector shards, and the benchmark resolves the same budget
and shard plan as a real run. A row is labelled when the outer pool is capped.
These are in-memory runs that pickle agents every round; a real run sends only
the sector description (workers load from PostgreSQL), so it has less IPC to do.

End to end, with PostgreSQL and spawn/insert included, 3000 agents (seed 7,
auto sectoring) take **2.8 s single-threaded and 1.3 s on 6 workers**, and the
run ends with one side at zero. Bigger sectors do more work per tick, but the
run needs far fewer rounds, so the total stays in the same ballpark as the old
finely split runs. The worker start-up is hidden behind the insert (see below).

### Overlapping start-up

Process pools cost about half a second to start on Windows. `Simulation`
therefore exposes `prewarm()`, which the UI calls *before* writing the spawns:
the workers spawn and open their PostgreSQL connections while the insert is
running, so the first round starts with a live pool. On the reference 3000-agent
run that cut the pooled time from 2.2 s to 1.2 s when it was introduced.

### What it costs

- Worker processes use ~30–50 MB each plus a one-off start-up of ~0.5 s on
  Windows; a run pays that at start-up, overlaps it with the spawn insert and
  reuses the pool until the living population halves. Shrinking pays the
  start-up again but frees the idle workers and their connections.
- Parallel width is `min(MAX_WORKERS, sectors)`, and sectors come from the
  population. 3000 agents therefore use at most 6 workers; raising
  `MAX_WORKERS` (or lowering `AGENTS_PER_SECTOR`) buys more cores at the cost of
  smaller arenas and fewer interactions between the two sides.
- Nearest-enemy search is grid-accelerated (expected O(1) per query), so a
  sector's per-tick cost grows roughly linearly with its agents instead of
  quadratically. `AGENTS_PER_SECTOR` is a target, but mono-type merging can make
  one sector larger (or the whole map one arena); bigger arenas mean fewer
  cross-sector rounds, not a scan bound.
- The accelerator adds one optional import and about 31 MB of installed files
  (`numpy`), plus about 10 MB for `wgpu` when installed. numpy costs a few
  milliseconds at import and nothing per run after that; a GPU run pays ~1 s of
  driver start-up on first use, holds one device for the whole process, and
  cannot use the worker pool.
- After each round the orchestrator clusters from the states the sector visits
  already reported (`SectorOutcome.final_states`) instead of issuing another
  `SELECT` over every living agent. The cache applies the same map clamping and
  integer rounding as `save_agents`, so it is byte-identical to a real re-read;
  if any sector fails to report, the next round falls back to PostgreSQL.
- Clustering stays serial; the benchmark prints its cost so it is never
  mistaken for a worker bottleneck.
- Headless mode captures no frames at all, so it is the fastest way to run
  (and the right mode for benchmarks); HTML (`--mode html`) records a sampled
  frame set for the player (every `FRAME_INTERVAL` ticks, capped by
  `MAX_FRAMES` and `MAX_FRAME_RECORDS`, with the first and last frame kept).
- Each worker holds its own PostgreSQL connection (at most `MAX_WORKERS`) and
  sectors write disjoint rows, so workers never contend for a row lock.

---

## Tests

```
.\venv\Scripts\python.exe -m unittest discover -s tests
```

The data-layer tests are skipped by default. Run the full pre-ship check
(unit suite + live PostgreSQL) with:

```
.\venv\Scripts\python.exe tools\check.py
```

Or run only the data-layer tests against a scratch PostgreSQL with:

```
.\tools\test_db.ps1
```

---

## Project layout

```
main.py                     menu, prompts, run orchestration
.gitignore                  ignores venv, caches, output and database.ini
config/
  database.ini              <-- PostgreSQL login: edit this file
  db_config.py              reads/writes database.ini, builds psycopg params
  simulation_config.py      every tuning knob (combat, spawning, sectors, visuals)
db/
  schema.py                 create/drop/clear the database and tables
  db_manager.py             load/insert/save agents, bulk writes, status counts
runtime/
  accelerator.py            nearest-enemy backends: python / numpy / WebGPU
  capabilities.py           CPU/GIL probes, worker and backend resolution
  sector_pool.py            process / thread / serial execution of sectors
src/
  agent.py                  Agent base class (movement, damage, bounds)
  human.py                  Human subclass (weapon_tier)
  zombie.py                 Zombie subclass (strength)
  spawner.py                random stat/position generation
  sector_engine.py          one sector's ticks: plan phases + serial apply
  sector_task.py            picklable worker task: load -> simulate -> save
  tick_executor.py          serial / thread executor for per-tick chunks
  simulation.py             round loop, scheduling, reports
utils/
  combat.py                 hit chance and damage formulas
  sectoring.py              grid sectoring and the sector-count policy
  spatial.py                uniform-grid nearest-enemy index
  spatial_batch.py          tiled numpy argmin for a whole group at once
vis/
  html_vis.py               self-contained HTML player
tools/
  bench_parallel.py         worker/backend speed-up benchmark
  check.py                  full check: unit suite + live PostgreSQL
  test_db.ps1               opt-in PostgreSQL test runner
tests/
  test_accel.py             backend agreement, f32 safety band, determinism
  test_parallel.py          sector policy, determinism, backend equivalence
  test_event_parallel.py    tick sharding, chunk size, frame interval, cache
  test_spatial.py           spatial index vs brute-force reference
  test_movement.py          humans hold at weapon range, simultaneous damage
  test_super_agent.py       super-agent rolls, doubled stats, triple health
  test_map_bounds.py        config/schema/spawner map bounds
  test_html_vis.py          frame cap, record budget, prune on finish
  test_main.py              CLI cap, exit codes, run/reset ordering, EOF exit
  test_db_config.py         blank-password handling and INI warnings
  test_schema.py            ensure_database fast path and CREATE fallback
  test_db_integration.py    opt-in real-PostgreSQL data-layer checks
output/                     generated HTML runs (created on demand)
```

---

## Tuning

Everything is in `config/simulation_config.py` – no other file needs editing:

```python
# Map: minimum half-extent, size step and target area per starting agent.
MAP_MIN_HALF_EXTENT = 100
MAP_SIZE_STEP = 100
MAP_AREA_PER_AGENT = 1000

# Sectoring: one sector per AGENTS_PER_SECTOR living agents, with the ceiling
# derived from the starting population.
AGENTS_PER_SECTOR = 500
SECTOR_PADDING = 25.0
SECTOR_COUNT = 0              # 0 = population rule; N pins the split
COLLAPSE_ON_STAGNATION = True # stalled round -> one global sector, keep going

# Parallelism
WORKERS = 0                   # 0 = one per usable CPU (capped by MAX_WORKERS)
MAX_WORKERS = 12              # real width is min(MAX_WORKERS, sectors)
WORKER_BACKEND = "auto"       # auto | serial | thread | process
EVENT_PARALLEL = "auto"       # auto | off | thread
EVENT_SHARDS = 0              # per-sector tick shards; 0 = split the budget
EVENT_CHUNK_SIZE = 0          # agents per chunk cap; 0 = even split
FRAME_INTERVAL = 2            # capture every Nth tick (1 = every tick)

# Nearest-enemy acceleration
ACCEL = "auto"                # auto | off | numpy | gpu
NUMPY_TILE_ELEMS = 1 << 18    # pairs per temporary numpy tile (memory bound)
ACCEL_MIN_PAIRS = 300         # below this, the scalar loop wins on any backend

# Loop control (all four are what stop the run from freezing)
SECTOR_REFRESH_TICKS = 15     # re-cluster and rebuild sectors after this many ticks
STALL_PATIENCE = 8            # stop a visit if the closest pair stops closing
MAX_ROUNDS = 400
MAX_ITERATIONS_PER_SECTOR = 2000
MAX_TOTAL_TICKS = MAX_ROUNDS * max(1, min(SECTOR_REFRESH_TICKS, MAX_ITERATIONS_PER_SECTOR))  # 6000 by default
NO_COMBAT_PATIENCE = 5

ZOMBIE_ATTACK_RANGE = 2.5
ZOMBIE_STOP_DISTANCE = 1.0
ZOMBIE_DAMAGE_PER_STRENGTH = (5, 15)

HUMAN_RANGE_PER_TIER = 4
HUMAN_DAMAGE_PER_TIER = (3, 9)
HUMAN_HIT_BASE = 0.9
HUMAN_HIT_FALLOFF = 0.045
HUMAN_HIT_FLOOR = 0.05

HUMAN_HEALTH_RANGE = (80, 150)
HUMAN_SPEED_RANGE = (1, 6)
HUMAN_WEAPON_TIER_RANGE = (1, 5)
ZOMBIE_HEALTH_RANGE = (50, 200)
ZOMBIE_SPEED_RANGE = (2, 5)
ZOMBIE_STRENGTH_RANGE = (2, 5)

SUPER_AGENT_CHANCE = 0.00002  # per spawned agent, either side
SUPER_STAT_MULTIPLIER = 2     # tier / strength above the spawn max
SUPER_HEALTH_MULTIPLIER = 3   # health above the spawn max
SUPER_SPEED = 3               # fixed, not doubled

MAX_AGENTS = 100000           # hard cap on one run's population
MAX_FRAMES = 2000
```

Note: `weapon_tier` and `strength` only have a `>= 1` schema check now; the
ranges above shape normal spawns, and super agents deliberately exceed them.
The health and speed ranges only shape new spawns.

### Map size

The map is chosen once per run from the starting population:

```
raw  = max(MAP_MIN_HALF_EXTENT,
           ceil(sqrt(total_agents * MAP_AREA_PER_AGENT) / 2))
half = raw snapped to the nearest MAP_SIZE_STEP
```

Area grows with the population, so average spacing (and how long a chase takes)
stays roughly constant. The extent is always a whole number on a 100-unit grid:
`-2700…2700`, then `-2800…2800`. The smallest map is `-100…100` on each axis;
there is no upper limit. The HTML player records the run's actual extent and
grid step, so it cannot drift out of sync.

Older databases may still carry the fixed `-1000…1000` `CHECK` constraints.
Bootstrap drops those constraints (leaving every agent row untouched) because a
static boundary cannot describe a per-run map. The same cleanup drops the old
bounded `1–5` stat checks, so super agents (tier/strength 10) insert into an
existing database without a reset. Bounds are enforced while spawning, saving
and moving agents.

### Command line

The menu is still the default, but the same run can be started directly — handy
for benchmarks and for reproducing a run:

```
python main.py --run --humans 1500 --zombies 1500 --seed 7 --headless
python main.py --run --humans 1500 --zombies 1500 --seed 7 --mode html --workers 8
python main.py --run --humans 1500 --zombies 1500 --seed 7 --sectors 3 --workers 1
python main.py --run --humans 200 --zombies 200 --seed 7 --backend thread --event-shards 4
```

| Flag | Meaning |
| --- | --- |
| `--run` | skip the menu, start immediately |
| `--humans`, `--zombies`, `--seed` | population and seed for `--run` |
| `--mode html\|headless` | HTML player (default) or summary only |
| `--headless` | shorthand for `--mode headless` |
| `--workers N` | worker threads/processes; `0` = auto |
| `--backend ...` | `auto`, `serial`, `thread` or `process` |
| `--event-shards N` | per-tick shards inside a sector; `0` = auto |
| `--sectors N` | pin the sector count (reproducible runs); `0` = auto |
| `--accel ...` | `auto`, `off`, `numpy` or `gpu` nearest-enemy backend |

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'psycopg'`** – the PostgreSQL driver is
not installed in the interpreter you are running. Fix it with:

```
pip install "psycopg[binary]"
```

This fails at import, before any menu or connection message, so it looks like a
crash rather than a missing dependency.

**"Could not reach PostgreSQL"** – the server is not running, or the
credentials are wrong. Use menu option 3, then option 4 to verify.

**Login file** – `config/database.ini` is the file to edit; menu option 3
rewrites it. If you delete it, it is recreated with default values on the next
run. Values left blank fall back to `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
`DB_PASSWORD`, then to `localhost:5432/zombie_sim` as `postgres`.

**One side gets wiped out** – that is the intended ending. Numbers dominate,
but humans now fire from full weapon range, so equal populations tend to favour
them. To shift the balance, raise `HUMAN_DAMAGE_PER_TIER` or `HUMAN_HIT_BASE`
(or lower `HUMAN_HIT_FALLOFF`) to favour humans, or raise
`ZOMBIE_DAMAGE_PER_STRENGTH` / `ZOMBIE_SPEED_RANGE` to favour zombies.
