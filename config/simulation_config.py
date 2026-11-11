"""Central tuning knobs for the zombie simulation.

Everything here can be edited without touching the simulation code.
"""

# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #
# The map is sized from the starting population so every agent gets roughly the
# same amount of space:
#
#     raw  = max(MAP_MIN_HALF_EXTENT,
#                ceil(sqrt(total_agents * MAP_AREA_PER_AGENT) / 2))
#     half = raw snapped to the nearest MAP_SIZE_STEP
#
# The snap keeps every map an integer box on a 100-unit grid (-2700..2700,
# -2800..2800, ...). The minimum keeps tiny runs in a usable arena; there is no
# upper limit.
MAP_MIN_HALF_EXTENT = 100
MAP_SIZE_STEP = 100
MAP_AREA_PER_AGENT = 1000

# --------------------------------------------------------------------------- #
# Sectoring
# --------------------------------------------------------------------------- #
# The map is split into one sector per this many living agents:
#
#     max_sectors = max(1, ceil(initial_agents / AGENTS_PER_SECTOR))
#     sectors     = clamp(ceil(living_agents / AGENTS_PER_SECTOR),
#                         1, max_sectors)
#
# 200 agents -> 1 sector, 1500 -> 3, 3000 -> 6. Fewer, larger sectors are what
# let humans and zombies actually meet: a finely split map can leave whole
# sectors with only one type, which is how a run used to end with both sides
# still alive. There is no fixed maximum: the cap scales with the starting
# population, and the actual count falls as agents die.
AGENTS_PER_SECTOR = 500
# A sector's walkable bounding box is the cluster's bounding box expanded by
# this many map units (clamped to the map). Agents never leave their sector.
SECTOR_PADDING = 25.0

# 0 = sectors follow the population rule above. Any other value pins the count
# (bounded by the run's population-derived maximum), which is how a run is
# reproduced exactly.
SECTOR_COUNT = 0

# If a round ends with both sides alive and nothing happened at all (no attack,
# no movement), the next round is run as one single sector instead of stopping
# the simulation. With both types in one arena every tick either lands a hit or
# closes a distance, so the run can only finish when one side is wiped out.
COLLAPSE_ON_STAGNATION = True

# --------------------------------------------------------------------------- #
# Parallelism
# --------------------------------------------------------------------------- #
# 0 = auto: one worker per usable CPU (affinity/cgroup aware), capped by
# MAX_WORKERS. Set to 1 to force the old single-core behaviour.
WORKERS = 0
# Sector count caps parallel width, so there is no point going wider than the
# number of sectors the population produces.
MAX_WORKERS = 12

# "auto"   -> processes on normal CPython (the GIL blocks parallel threads),
#             threads on free-threaded builds, single-threaded when 1 CPU.
# "serial" -> never use a pool.
# "thread" -> force threads (useful for profiling / DB-bound runs).
# "process"-> force a process pool.
WORKER_BACKEND = "auto"

# Per-tick event parallelism (inside a sector).  Only meaningful when the outer
# backend is "thread": on the GIL a worker thread cannot speed up the CPU loop,
# so the default leaves it off there.  "auto" enables it whenever the resolved
# outer backend is thread.
# "auto" | "off" | "thread"
EVENT_PARALLEL = "auto"
# Shards per sector; 0 = auto (fill the worker budget without oversubscribing).
# The result never depends on the shard count: proposals are merged in
# canonical order and all RNG draws stay in the merge.
EVENT_SHARDS = 0
# Optional cap on agents per shard; 0 = let the engine split evenly.
EVENT_CHUNK_SIZE = 0
# Capture a visualisation frame every N ticks (1 = every tick).  Higher values
# shrink the HTML payload for long runs.
FRAME_INTERVAL = 2

# --------------------------------------------------------------------------- #
# Acceleration (nearest-enemy query)
# --------------------------------------------------------------------------- #
# "auto"   -> numpy when it is installed, pure Python otherwise.  Auto stays on
#             the CPU because numpy works inside the process pool, while a GPU
#             run has to be single-threaded.
# "numpy"  -> force the vectorised CPU path.
# "gpu"    -> force WebGPU when the machine has a usable adapter; otherwise
#             fall back to numpy (or pure Python) automatically.
# "off"    -> always use the reference scalar loop.
ACCEL = "auto"

# Query pairs per temporary matrix tile.  Bounds numpy's peak memory to roughly
# 3 x 8 bytes x this many; the sector engine only ever asks for ~500x500.
NUMPY_TILE_ELEMS = 1 << 18

# Below this many query-target pairs the Python loop beats handing the work to
# numpy (measured crossover is ~300 pairs on a 12-core desktop).
ACCEL_MIN_PAIRS = 300

# --------------------------------------------------------------------------- #
# Simulation loop
# --------------------------------------------------------------------------- #
# A sector runs for at most this many ticks before the map is re-clustered and
# the sectors are rebuilt from the agents' CURRENT positions. Lower = sectors
# follow the action more closely, higher = fewer database round trips.
SECTOR_REFRESH_TICKS = 15

# Hard safety nets. A run can never exceed these, so it can never hang.
MAX_ROUNDS = 400

# Absolute per-sector-visit cap (never exceeded even if SECTOR_REFRESH_TICKS is
# raised above it).
MAX_ITERATIONS_PER_SECTOR = 2000


# Derived so raising MAX_ROUNDS also raises the tick budget. A round advances
# world time by at most min(SECTOR_REFRESH_TICKS, MAX_ITERATIONS_PER_SECTOR).
def _tick_budget(refresh_ticks: int) -> int:
    """Whole-run tick cap; a zero refresh still advances one tick per round."""
    visit_ticks = max(1, min(refresh_ticks, MAX_ITERATIONS_PER_SECTOR))
    return MAX_ROUNDS * visit_ticks


MAX_TOTAL_TICKS = _tick_budget(SECTOR_REFRESH_TICKS)

# Stop a sector when the closest human/zombie pair has not got any closer for
# this many ticks. This is what prevents a sector that never engages from
# grinding on forever - it works whether or not combat has started.
STALL_PATIENCE = 8

# After combat has started, stop once this many consecutive iterations pass
# without a single attack ("until no combat happens").
NO_COMBAT_PATIENCE = 5

# --------------------------------------------------------------------------- #
# Combat
# --------------------------------------------------------------------------- #
# Zombies: fixed reach, no accuracy roll. Damage is strength-scaled.
ZOMBIE_ATTACK_RANGE = 2.5
ZOMBIE_STOP_DISTANCE = 1.0
ZOMBIE_DAMAGE_PER_STRENGTH = (5, 15)  # randint(strength*5, strength*15)

# Humans: reach scales with weapon tier, damage is random per tier and shots
# can miss - the farther away the target, the lower the hit chance. A human
# walks until the target is inside its range and then holds there, firing.
HUMAN_RANGE_PER_TIER = 2.5          # tier 6 -> 15 units
HUMAN_DAMAGE_PER_TIER = (3, 9)       # randint(tier*3, tier*9)
HUMAN_HIT_BASE = 0.9                # point-blank accuracy
HUMAN_HIT_FALLOFF = 0.045            # accuracy lost per unit per tier
HUMAN_HIT_FLOOR = 0.05

# --------------------------------------------------------------------------- #
# Spawn stats
# --------------------------------------------------------------------------- #
HUMAN_HEALTH_RANGE = (80, 150)
HUMAN_SPEED_RANGE = (1, 6)
HUMAN_WEAPON_TIER_RANGE = (1, 6)

ZOMBIE_HEALTH_RANGE = (50, 200)
ZOMBIE_SPEED_RANGE = (2, 5)
ZOMBIE_STRENGTH_RANGE = (2, 5)

# Super agents: rare individuals of either side whose weapon tier / strength is
# double the spawn maximum and whose health is triple it. Speed is fixed rather
# than doubled. The chance is rolled once per spawned agent, so it is also the
# knob to raise temporarily to see one in a small run.
SUPER_AGENT_CHANCE = 0.00002
SUPER_STAT_MULTIPLIER = 2
SUPER_HEALTH_MULTIPLIER = 3
SUPER_SPEED = 3

# Hard cap on one run's population. The spawn list is built in memory before
# any database write, so a typo must not be able to exhaust the machine.
MAX_AGENTS = 100_000

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #
MAX_FRAMES = 2000            # safety cap for HTML frame capture
# Total agent records kept across all frames. When exceeded, the player keeps
# every other frame instead of dropping the middle of the run.
# shortcut: 1M records trades detail for memory; raise it for a richer player.
MAX_FRAME_RECORDS = 1_000_000
HTML_OUTPUT_DIR = "output"
