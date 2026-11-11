"""Accelerated nearest-enemy search: GPU first, numpy second, pure Python last.

The sector engine spends most of its time answering "who is the closest living
enemy?" for every agent, every tick.  That question is one batched operation per
group, so it can be answered by a compute shader or by numpy instead of a
Python loop.

Backends, in the order they are tried:

``gpu``    one WebGPU compute dispatch (``wgpu-py``).  WebGPU is used because it
           is the only API that covers AMD, Intel, NVIDIA and Apple hardware on
           Windows, macOS and Linux from a single dependency, and its wheels are
           ``py3-none`` so they do not pin the Python version.
``numpy``  the same query in exact float64 on the CPU.
``python`` :class:`utils.spatial.SpatialIndex`, the reference path.

Determinism is not negotiable: a seeded run must produce the same rows no matter
which backend ran it.  ``f32`` cannot reproduce float64 tie-breaking, so the GPU
is only trusted to produce *candidates*; the winner is always confirmed on the
CPU in float64 (:func:`exact_distances`), and a candidate is only accepted when
the runner-up is clearly farther away.  Ambiguous queries are recomputed
exactly.  A backend therefore cannot change a result, only the speed.

Everything degrades instead of failing: no ``wgpu`` module, no adapter, a
software adapter, a shader error or a device lost mid-run all fall back to
numpy/python and log once.
"""

from __future__ import annotations

import math
import threading
import time

from config import simulation_config as cfg

VALID_MODES = ("auto", "off", "numpy", "gpu")

# Targets whose squared distance is inside the float32 error band of the best
# one are re-checked exactly.  See GpuAccelerator._gpu_indices for the bound.
_F32_EPS = 2.0**-23

# Adapter names that mean "there is no real GPU here".  wgpu can expose a
# software rasteriser through D3D12/Vulkan, and using it would be slower than
# numpy while claiming to be a GPU run.
_SOFTWARE_HINTS = (
    "basic render",
    "software",
    "llvmpipe",
    "lavapipe",
    "swiftshader",
    "warp",
)

# wgpu's OpenGL backend has no usable compute/storage support; skip it.
_SKIP_BACKENDS = ("OpenGL", "OpenGLES")

# One instance per query: each thread scans its group's targets in order.  A
# workgroup-per-query reduction would need six barriers to scan ~250 targets,
# which costs more than the scan itself at this sector size.
_SHADER = """
struct Result {
  best: u32,
  second: u32,
  best_d2: f32,
  second_d2: f32,
};

@group(0) @binding(0) var<uniform> counts: vec2<u32>;
@group(0) @binding(1) var<storage, read> queries: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> targets: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> results: array<Result>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= counts.x) {
    return;
  }
  let q = queries[i];
  var best: f32 = 3.402823e38;
  var second: f32 = 3.402823e38;
  var best_i: u32 = 0xFFFFFFFFu;
  var second_i: u32 = 0xFFFFFFFFu;

  for (var j: u32 = 0u; j < counts.y; j = j + 1u) {
    let d = targets[j] - q;
    let d2 = d.x * d.x + d.y * d.y;
    if (d2 < best) {
      second = best;
      second_i = best_i;
      best = d2;
      best_i = j;
    } else if (d2 < second) {
      second = d2;
      second_i = j;
    }
  }

  results[i] = Result(best_i, second_i, best, second);
}
"""


def exact_distances(queries, targets, indices) -> list[float]:
    """Turn chosen indices into the distances the scalar path would report.

    ``math.sqrt`` on the same float64 expression as ``SpatialIndex.nearest``
    keeps every downstream decision (``can_reach``, ``hit_chance``,
    ``planned_move``) bit-identical across backends.
    """
    out: list[float] = []
    for (qx, qy), index in zip(queries, indices):
        if index < 0:
            out.append(float("inf"))
            continue
        tx, ty = targets[index]
        dx = tx - qx
        dy = ty - qy
        out.append(math.sqrt(dx * dx + dy * dy))
    return out


def _needs_exact(best, second, scale):
    """Is the f32 runner-up too close to the winner to trust?

    f32 error is dominated by *coordinate* magnitude, not by the distance:
    subtracting two large f32 positions loses precision.  With ``u = 2^-23``
    and ``S`` the largest coordinate in the batch, a squared distance can be
    off by about ``12*S*u*sqrt(d2) + 4*u*d2``, so anything within that band is
    recomputed in float64.  Works on floats (tests) and numpy arrays (dispatch).
    """
    band = 12.0 * scale * _F32_EPS * (best ** 0.5) + 4.0 * _F32_EPS * best
    return second - best <= band


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class PythonAccelerator:
    """Present so callers can treat "no acceleration" as a normal backend."""

    name = "python"
    detail = "pure Python"

    def nearest_cross(self, queries, targets):
        if not queries or not targets:
            return [-1] * len(queries), [float("inf")] * len(queries)
        # Kept simple on purpose: this path only runs when numpy is missing.
        indices = []
        for qx, qy in queries:
            best_i, best_d2 = -1, float("inf")
            for index, (tx, ty) in enumerate(targets):
                dx = tx - qx
                dy = ty - qy
                d2 = dx * dx + dy * dy
                if d2 < best_d2:
                    best_i, best_d2 = index, d2
            indices.append(best_i)
        return indices, exact_distances(queries, targets, indices)


class NumpyAccelerator:
    name = "numpy"
    detail = "numpy float64"

    def __init__(self) -> None:
        from utils import spatial_batch

        self._batch = spatial_batch
        self._scalar = PythonAccelerator()

    def nearest_cross(self, queries, targets):
        if not queries or not targets:
            return [-1] * len(queries), [float("inf")] * len(queries)
        if len(queries) * len(targets) < int(cfg.ACCEL_MIN_PAIRS):
            # A handful of agents is cheaper to compare in Python than to hand
            # to numpy; this is also the end-game tail of most runs.
            return self._scalar.nearest_cross(queries, targets)
        indices = self._batch.nearest_indices(queries, targets)
        return indices, exact_distances(queries, targets, indices)


class GpuAccelerator:
    """WebGPU compute backend with a numpy safety net."""

    name = "gpu"

    def __init__(self, adapter, numpy_backend: NumpyAccelerator) -> None:
        import wgpu  # local import: the module is optional at runtime

        self._wgpu = wgpu
        self._numpy = numpy_backend
        info = adapter.info or {}
        self.detail = "{} ({})".format(
            info.get("device", "unknown GPU"),
            info.get("backend_type", "unknown backend"),
        )
        self._device = adapter.request_device_sync()
        self._queue = self._device.queue
        self._broken = ""
        self._sizes: tuple[int, int, int] | None = None
        self._buffers: dict = {}
        self._bind_group = None
        self._params_buffer = None
        self._pipeline = self._build_pipeline()

    # ------------------------------------------------------------------ #
    def _build_pipeline(self):
        wgpu = self._wgpu
        module = self._device.create_shader_module(code=_SHADER)
        layout = self._device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "uniform"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "read-only-storage"},
                },
                {
                    "binding": 3,
                    "visibility": wgpu.ShaderStage.COMPUTE,
                    "buffer": {"type": "storage"},
                },
            ]
        )
        self._layout = layout
        pipeline_layout = self._device.create_pipeline_layout(
            bind_group_layouts=[layout]
        )
        return self._device.create_compute_pipeline(
            layout=pipeline_layout,
            compute={"module": module, "entry_point": "main"},
        )

    # ------------------------------------------------------------------ #
    def _ensure_buffers(self, query_count: int, target_count: int) -> None:
        """Grow the four buffers to the largest group seen so far.

        A device is per-process, so the buffers are reused across every sector,
        tick and round instead of being reallocated per dispatch.
        """
        wgpu = self._wgpu
        usage = wgpu.BufferUsage
        sizes = (
            max(16, query_count * 8),
            max(16, target_count * 8),
            max(16, query_count * 16),
        )
        if self._sizes is not None and all(
            have >= need for have, need in zip(self._sizes, sizes)
        ):
            return
        for buffer in self._buffers.values():
            buffer.destroy()
        self._sizes = sizes
        self._buffers = {
            "queries": self._device.create_buffer(
                size=sizes[0], usage=usage.STORAGE | usage.COPY_DST
            ),
            "targets": self._device.create_buffer(
                size=sizes[1], usage=usage.STORAGE | usage.COPY_DST
            ),
            "results": self._device.create_buffer(
                size=sizes[2], usage=usage.STORAGE | usage.COPY_SRC
            ),
            "staging": self._device.create_buffer(
                size=sizes[2], usage=usage.MAP_READ | usage.COPY_DST
            ),
        }
        if self._params_buffer is None:
            self._params_buffer = self._device.create_buffer(
                size=16, usage=usage.UNIFORM | usage.COPY_DST
            )
        self._bind_group = self._device.create_bind_group(
            layout=self._layout,
            entries=[
                {"binding": 0, "resource": {"buffer": self._params_buffer}},
                {"binding": 1, "resource": {"buffer": self._buffers["queries"]}},
                {"binding": 2, "resource": {"buffer": self._buffers["targets"]}},
                {"binding": 3, "resource": {"buffer": self._buffers["results"]}},
            ],
        )

    # ------------------------------------------------------------------ #
    def nearest_cross(self, queries, targets):
        if self._broken:
            return self._numpy.nearest_cross(queries, targets)
        if not queries or not targets:
            return [-1] * len(queries), [float("inf")] * len(queries)
        if len(queries) * len(targets) < int(cfg.ACCEL_MIN_PAIRS):
            # A dispatch costs ~0.6 ms no matter how little work it does, so
            # the tail of a run (a handful of survivors) stays on the CPU.
            return self._numpy.nearest_cross(queries, targets)
        try:
            indices = self._gpu_indices(queries, targets)
        except Exception as exc:  # noqa: BLE001 - a GPU must never break a run
            self._broken = f"{type(exc).__name__}: {exc}"
            print(
                f"  [accel] GPU dispatch failed ({self._broken}); "
                "continuing on the CPU"
            )
            return self._numpy.nearest_cross(queries, targets)
        return indices, exact_distances(queries, targets, indices)

    # ------------------------------------------------------------------ #
    def _gpu_indices(self, queries, targets) -> list[int]:
        import numpy as np

        wgpu = self._wgpu
        query_count, target_count = len(queries), len(targets)
        self._ensure_buffers(query_count, target_count)
        query_buf = self._buffers["queries"]
        target_buf = self._buffers["targets"]
        result_buf = self._buffers["results"]
        staging = self._buffers["staging"]

        q = np.asarray(queries, dtype=np.float32).reshape(query_count, 2)
        t = np.asarray(targets, dtype=np.float32).reshape(target_count, 2)
        self._queue.write_buffer(query_buf, 0, q.tobytes())
        self._queue.write_buffer(target_buf, 0, t.tobytes())
        self._queue.write_buffer(
            self._params_buffer,
            0,
            np.asarray([query_count, target_count], dtype=np.uint32).tobytes()
            + b"\x00" * 8,
        )

        encoder = self._device.create_command_encoder()
        pass_ = encoder.begin_compute_pass()
        pass_.set_pipeline(self._pipeline)
        pass_.set_bind_group(0, self._bind_group)
        pass_.dispatch_workgroups((query_count + 63) // 64, 1, 1)
        pass_.end()
        encoder.copy_buffer_to_buffer(
            result_buf, 0, staging, 0, query_count * 16
        )
        self._queue.submit([encoder.finish()])

        staging.map_sync(mode=wgpu.MapMode.READ)
        try:
            raw = bytes(staging.read_mapped())
        finally:
            staging.unmap()

        results = np.frombuffer(
            raw,
            dtype=np.dtype(
                [("best", "<u4"), ("second", "<u4"), ("best_d2", "<f4"), ("second_d2", "<f4")]
            ),
            count=query_count,
        )
        indices = results["best"].astype(np.int64)

        # The GPU only nominated a winner.  Anything it could plausibly have
        # got wrong is decided exactly here, so results never depend on f32.
        #
        # f32 error is dominated by *coordinate* magnitude, not by the distance:
        # subtracting two large f32 positions loses precision.  With u = 2^-23
        # and S = the largest coordinate in the batch, the squared distance can
        # be off by about 12*S*u*sqrt(d2) + 4*u*d2; anything inside that band is
        # recomputed in float64.
        best = results["best_d2"].astype(np.float64)
        second = results["second_d2"].astype(np.float64)
        scale = max(
            float(np.abs(q).max(initial=0.0)),
            float(np.abs(t).max(initial=0.0)),
        )
        ambiguous = (results["second"] != 0xFFFF_FFFF) & _needs_exact(
            best, second, scale
        )
        if ambiguous.any():
            # Recompute only the doubtful queries exactly.  A fully ambiguous
            # batch costs the same as this (every query is doubtful), so there
            # is no separate "give up" path to take.
            flagged = np.nonzero(ambiguous)[0]
            exact = self._numpy.nearest_cross(
                [queries[k] for k in flagged], targets
            )[0]
            for k, index in zip(flagged, exact):
                indices[k] = index
        return indices.tolist()


# --------------------------------------------------------------------------- #
# Probing and the process-wide singleton
# --------------------------------------------------------------------------- #
class AccelPlan:
    """The resolved acceleration choice for one run."""

    __slots__ = ("mode", "detail", "calibrate")

    def __init__(self, mode: str, detail: str = "", calibrate: bool = False) -> None:
        self.mode = mode
        self.detail = detail
        # Only a GPU request calibrates: it may demote the device after
        # measuring it against numpy.
        self.calibrate = calibrate

    @property
    def is_accelerated(self) -> bool:
        return self.mode in ("numpy", "gpu")

    def describe(self) -> str:
        if self.mode == "gpu":
            return f"gpu ({self.detail})"
        if self.mode == "numpy":
            return "numpy (CPU)"
        return "pure Python (scalar grid)"


def _numpy_available() -> bool:
    try:
        import numpy  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - a broken install is the same as absent
        return False


def _pick_adapter(readable):
    """Choose the best real adapter, or None.

    Discrete/AMD/NVIDIA/Apple parts are preferred, integrated ones are accepted,
    and software rasterisers and wgpu's GL backend are rejected: they would be
    slower than numpy while reporting a "gpu" backend.
    """
    try:
        adapters = list(readable)
    except Exception:  # noqa: BLE001
        return None, "no adapters"

    best = None
    best_score = -1
    for adapter in adapters:
        info = getattr(adapter, "info", None) or {}
        device = str(info.get("device") or "")
        backend = str(info.get("backend_type") or "")
        if backend in _SKIP_BACKENDS:
            continue
        lowered = device.lower()
        if any(hint in lowered for hint in _SOFTWARE_HINTS):
            continue
        # Vulkan/Metal report capability honestly; D3D12 is fine too.
        score = 2 if backend in ("Vulkan", "Metal") else 1
        if any(word in lowered for word in ("nvidia", "geforce", "radeon", "arc ")):
            score += 1
        if score > best_score:
            best, best_score = adapter, score
    if best is None:
        return None, "no usable adapter"
    info = best.info or {}
    return best, str(info.get("device") or "unknown GPU")


_global_lock = threading.RLock()
_global_accel = None
_global_mode = ""
_global_requested = ""
_global_plan: AccelPlan | None = None
_adapter_cache: tuple | None = None


def _gpu_adapter():
    """``(adapter, description)``; ``(None, reason)`` when there is no GPU."""
    global _adapter_cache
    if _adapter_cache is not None:
        return _adapter_cache
    try:
        import wgpu
    except Exception as exc:  # noqa: BLE001 - missing/broken wgpu is not an error
        _adapter_cache = (None, f"wgpu unavailable ({type(exc).__name__})")
        return _adapter_cache
    try:
        _adapter_cache = _pick_adapter(wgpu.gpu.enumerate_adapters_sync())
    except Exception as exc:  # noqa: BLE001
        _adapter_cache = (None, f"adapter probe failed ({type(exc).__name__})")
    return _adapter_cache


def _calibrate(gpu, numpy_backend) -> bool:
    """Is the GPU actually faster than numpy for this engine's workload?

    The engine asks about a few hundred agents at a time, where a dispatch can
    cost more than the arithmetic.  Measuring beats guessing: a machine that
    says "gpu" in its spec sheet but loses to numpy here keeps numpy.
    """
    import random

    rng = random.Random(1234)
    queries = [(rng.uniform(-500, 500), rng.uniform(-500, 500)) for _ in range(250)]
    targets = [(rng.uniform(-500, 500), rng.uniform(-500, 500)) for _ in range(250)]
    repeats = 10

    def timed(backend) -> float:
        backend.nearest_cross(queries, targets)  # compile/cache warm-up
        started = time.perf_counter()
        for _ in range(repeats):
            backend.nearest_cross(queries, targets)
        return time.perf_counter() - started

    try:
        numpy_time = min(timed(numpy_backend) for _ in range(2))
        gpu_time = min(timed(gpu) for _ in range(2))
    except Exception:  # noqa: BLE001 - a failing GPU is simply not chosen
        return False
    return gpu_time < numpy_time


def probe(requested: str | None = None) -> AccelPlan:
    """Resolve ``auto|off|numpy|gpu`` into a mode without creating a device."""
    raw = (cfg.ACCEL if requested is None else requested) or "auto"
    raw = str(raw).strip().lower()
    if raw not in VALID_MODES:
        raw = "auto"

    if raw in ("off", "python", "none", "false", "0"):
        return AccelPlan("python", "disabled")

    has_numpy = _numpy_available()
    if raw in ("auto", "numpy"):
        # ``auto`` stays on numpy without touching the GPU stack: enumerating
        # adapters costs about a second of driver start-up, and numpy composes
        # with the process pool while a GPU run has to be single-threaded.
        # ``--accel gpu`` is the explicit "use my GPU" request.
        return AccelPlan("numpy" if has_numpy else "python")

    if not has_numpy:
        # The GPU's f32 result always needs a float64 safety net.
        return AccelPlan("python", "numpy missing")

    adapter, detail = _gpu_adapter()
    if adapter is None:
        return AccelPlan("numpy", detail)
    return AccelPlan("gpu", detail, calibrate=True)


def get() -> object:
    """The process-wide accelerator, created once (a GPU device is not free)."""
    global _global_accel, _global_mode
    with _global_lock:
        if _global_accel is None:
            if _global_mode == "gpu":
                numpy_backend = NumpyAccelerator()
                try:
                    adapter, detail = _gpu_adapter()
                    if adapter is None:
                        raise RuntimeError(detail)
                    candidate = GpuAccelerator(adapter, numpy_backend)
                    if (
                        _global_plan is not None
                        and _global_plan.calibrate
                        and not _calibrate(candidate, numpy_backend)
                    ):
                        print(
                            f"  [accel] {candidate.detail} is slower than numpy "
                            "for this workload; using numpy"
                        )
                        _global_mode = "numpy"
                        _global_accel = numpy_backend
                    else:
                        _global_accel = candidate
                except Exception as exc:  # noqa: BLE001 - degrade, never abort
                    print(
                        f"  [accel] GPU unavailable "
                        f"({type(exc).__name__}: {exc}); using numpy"
                    )
                    _global_accel = numpy_backend
                    _global_mode = "numpy"
            elif _global_mode == "numpy":
                _global_accel = NumpyAccelerator()
            else:
                _global_accel = PythonAccelerator()
        return _global_accel


def mode() -> str:
    """The effective backend right now ("python" until configured)."""
    return _global_mode or "python"


def requested() -> str:
    """The mode this process was configured with ("auto" until configured)."""
    return _global_requested


def configure(requested: str | None = None) -> AccelPlan:
    """Fix the backend for this process.  Safe to call more than once."""
    global _global_mode, _global_accel, _global_plan, _global_requested
    plan = probe(requested)
    raw = (cfg.ACCEL if requested is None else requested) or "auto"
    key = str(raw).strip().lower()
    with _global_lock:
        if _global_plan is not None and key == _global_requested:
            return _global_plan
        close()
        _global_requested = key
        _global_mode = plan.mode
        _global_plan = plan
    return plan


def close() -> None:
    """Drop the device (idempotent); the next ``get()`` rebuilds it."""
    global _global_accel
    with _global_lock:
        _global_accel = None

