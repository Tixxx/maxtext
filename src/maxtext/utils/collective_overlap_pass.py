"""Profile-guided POST_SCHEDULER pass: move async collective starts earlier.

Phase 1 — schedule reordering:
  For every async collective found in the scheduled HLO, use the PGLE FDO
  profile to check whether ops between start and done are sufficient to fully
  hide the collective's latency.  If not, move the start earlier (bounded by
  data-dependency constraints).

Phase 2 — split batched collectives:
  If a collective still has a latency deficit after phase 1 (because it bundles
  gradients from multiple layers and can't move due to deps), split it into
  per-layer sub-collectives positioned right after their respective producers,
  so each sub-collective overlaps with the next layer's backward GEMMs.

Registration
------------
Call ``register()`` once before the first ``jax.jit``-compiled function runs.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _logger.addHandler(_h)
    _logger.propagate = False

# ---------------------------------------------------------------------------
# Shared mutable state: populated once PGLE delivers its FDO profile bytes
# ---------------------------------------------------------------------------
_profile_costs: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Unified proto compilation (PGLE profile + XLA HLO module)
# All protos compiled into one directory to avoid descriptor pool conflicts.
# ---------------------------------------------------------------------------
_XLA_SRC = os.environ.get("DEFAULT_XLA_PATH") or "/opt/xla"
_TSL_SRC = f"{_XLA_SRC}/third_party/tsl"
_PROTO_OUT_DIR = "/tmp/_collective_overlap_pass_proto"

_PROTO_SOURCES = [
    "tsl/profiler/protobuf/profiled_instructions.proto",
    "tsl/profiler/protobuf/xplane.proto",
    "xla/service/hlo.proto",
    "xla/xla_data.proto",
    "xla/service/metrics.proto",
]

_PROTO_INIT_DIRS = [
    "",
    "tsl", "tsl/profiler", "tsl/profiler/protobuf",
    "xla", "xla/service",
]


def _ensure_protos():
    """Compile all needed proto files once into a single output directory."""
    marker = os.path.join(_PROTO_OUT_DIR, "xla", "service", "hlo_pb2.py")
    if os.path.exists(marker):
        if _PROTO_OUT_DIR not in sys.path:
            sys.path.insert(0, _PROTO_OUT_DIR)
        return
    for rel in _PROTO_INIT_DIRS:
        d = os.path.join(_PROTO_OUT_DIR, rel)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "__init__.py"), "a").close()
    subprocess.run(
        ["protoc",
         f"--proto_path={_TSL_SRC}",
         f"--proto_path={_XLA_SRC}",
         "--proto_path=/usr/local/include",
         f"--python_out={_PROTO_OUT_DIR}",
         *_PROTO_SOURCES],
        check=True, capture_output=True,
    )
    if _PROTO_OUT_DIR not in sys.path:
        sys.path.insert(0, _PROTO_OUT_DIR)


def _load_costs_from_fdo(fdo_bytes: bytes) -> dict[str, float]:
    _ensure_protos()
    from tsl.profiler.protobuf import profiled_instructions_pb2 as _pi  # type: ignore
    profiled = _pi.ProfiledInstructionsProto()
    profiled.ParseFromString(fdo_bytes)
    return {c.name: c.cost_us for c in profiled.costs}


def _update_profile(fdo_bytes: bytes) -> None:
    global _profile_loaded
    if not fdo_bytes:
        return
    try:
        costs = _load_costs_from_fdo(fdo_bytes)
        if not costs:
            return
        _profile_costs.update(costs)
        ag_count = sum(1 for k in costs if "all-gather" in k or "reduce-scatter" in k)
        if ag_count > 0:
            _profile_loaded = True
            _logger.info(
                "collective_overlap_pass: loaded PGLE profile with %d instruction "
                "costs (%d collective entries); total pool now %d.",
                len(costs), ag_count, len(_profile_costs),
            )
        else:
            _logger.debug(
                "collective_overlap_pass: merged profile chunk: %d entries, "
                "0 collectives; total pool now %d.",
                len(costs), len(_profile_costs),
            )
        _apply_te_ep_cost_overrides()
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning("collective_overlap_pass: failed to parse FDO profile: %s", exc)


# ---------------------------------------------------------------------------
# te_ep_* cost correction from a reference profiler trace
# ---------------------------------------------------------------------------
# PGLE's xplane->FDO conversion averages cost_us across every GPU kernel
# launch tagged with an HLO instruction's name. That's correct for a
# single-kernel op, but TE's expert-parallel dispatch/combine/prepare
# custom-calls are each actually ~7 sequential GPU kernels (Memsets, a
# prob-density conversion, the NCCL kernel itself, a local-permute, etc.)
# sharing one HLO name -- PGLE's mean ends up close to the per-kernel
# average, not the per-invocation total. Empirically confirmed: PGLE
# reports ~459us for te_ep_dispatch_ffi.10 where the real per-invocation
# wall-clock span is ~3300us (~7x underestimate; other te_ep_* families
# show 2x-22x depending on how many tiny sub-kernels dilute the mean).
# Rather than special-case each family, this loads real per-invocation
# costs from a REFERENCE profiler trace (pointed to by
# COLLECTIVE_OVERLAP_REFERENCE_TRACE) and overrides PGLE's te_ep_* entries.
#
# Must be a *reference* trace, not this run's own: the pass runs during
# compilation, before this run has produced any profiler capture of its
# own -- same two-pass (profile once, point later runs at it) workflow PGLE
# itself needs.
_TE_EP_PREFIX = "te_ep"
_INVOCATION_GAP_US = 100.0  # gap between GPU-stream events that starts a new invocation
_te_ep_overrides: dict[str, float] = {}
_te_ep_overrides_loaded = False


def _load_te_ep_costs_from_trace(trace_path: str) -> dict[str, float]:
    """Compute real per-invocation te_ep_* costs from a profiler trace.

    Parses a Chrome-trace-format trace.json[.gz] (the same format the
    xplane->trace.json.gz conversion produces, e.g. under
    <run>/tensorboard/plugins/profile/<timestamp>/<host>.trace.json.gz),
    filters to GPU:0's collective/NCCL stream, groups events by their hlo_op
    tag, clusters consecutive events into invocations (a new invocation
    starts whenever the gap since the previous event's end exceeds
    _INVOCATION_GAP_US), sums each invocation's constituent kernel durations
    (since they run back-to-back implementing one logical async op), and
    averages those per-invocation sums across invocations. Returns
    {hlo_op_name: avg_invocation_cost_us} for every te_ep_*-prefixed name
    found.
    """
    import gzip  # pylint: disable=import-outside-toplevel
    import json  # pylint: disable=import-outside-toplevel

    try:
        opener = gzip.open if trace_path.endswith(".gz") else open
        with opener(trace_path, "rt") as f:
            trace = json.load(f)
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning(
            "collective_overlap_pass: failed to read reference trace %s: %s",
            trace_path, exc,
        )
        return {}

    by_op: dict[str, list[tuple[float, float]]] = {}
    for event in trace.get("traceEvents", ()):
        if event.get("ph") != "X" or event.get("pid") != 1 or event.get("tid") != 83:
            continue
        ts, dur = event.get("ts"), event.get("dur")
        if ts is None or dur is None or dur <= 0:
            continue
        args = event.get("args") or {}
        hlo_op = args.get("hlo_op") or args.get("long_name") or event.get("name", "")
        if not hlo_op.startswith(_TE_EP_PREFIX):
            continue
        by_op.setdefault(hlo_op, []).append((ts, dur))

    return _average_invocation_costs(by_op)


def _average_invocation_costs(by_op: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    """Given {hlo_op: [(ts_us, dur_us), ...]} GPU-stream events (in any
    order), cluster each op's events into invocations (a new invocation
    starts whenever the gap since the previous event's end within that op
    exceeds _INVOCATION_GAP_US), sum each invocation's constituent event
    durations (they run back-to-back implementing one logical async op),
    and average those per-invocation sums across invocations. Returns
    {hlo_op: avg_invocation_cost_us}."""
    costs: dict[str, float] = {}
    for hlo_op, events in by_op.items():
        if not events:
            continue
        events.sort()
        invocation_spans: list[float] = []
        group_start, group_end = events[0][0], events[0][0] + events[0][1]
        for ts, dur in events[1:]:
            if ts - group_end > _INVOCATION_GAP_US:
                invocation_spans.append(group_end - group_start)
                group_start = ts
            group_end = max(group_end, ts + dur)
        invocation_spans.append(group_end - group_start)
        costs[hlo_op] = sum(invocation_spans) / len(invocation_spans)
    return costs


def _load_te_ep_costs_from_xspace_bytes(xspace_bytes: bytes) -> dict[str, float]:
    """Compute real per-invocation te_ep_* costs directly from a raw,
    in-memory serialized XSpace protobuf (as returned by
    ProfilerSession.stop(), the same bytes PGLE's own
    ConvertXplaneToProfiledInstructionsProto consumes -- see
    _patch_pgle_profiler).

    This is the live, in-run counterpart to _load_te_ep_costs_from_trace:
    instead of reading a previously-written trace.json.gz from some earlier
    run, it parses PGLE's own profiling data for *this* run directly,
    before it gets discarded, so the correction is available in time for
    the recompile that immediately follows PGLE's profiling retries -- no
    external reference trace or bootstrapping run required.

    Walks XSpace -> XPlane (device planes only, name prefixed
    "/device:GPU:") -> XLine -> XEvent, resolving each event's "hlo_op" stat
    (checking both the event's own stats and its XEventMetadata's constant
    stats, matching xplane_to_profile_instructions.cc's GetXPlaneLatencyInfo)
    via either XStat.str_value directly or XStat.ref_value pointing at
    another XStatMetadata's name (the interned-string form). Event
    timestamps are computed as XLine.timestamp_ns*1000 + XEvent.offset_ps
    (picoseconds since epoch); only the relative ordering/spacing within an
    op matters here, so the absolute epoch offset is irrelevant.
    """
    _ensure_protos()
    from tsl.profiler.protobuf import xplane_pb2 as _xp  # type: ignore  # pylint: disable=import-outside-toplevel

    try:
        xspace = _xp.XSpace()
        xspace.ParseFromString(xspace_bytes)
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning(
            "collective_overlap_pass: failed to parse live XSpace profile: %s", exc,
        )
        return {}

    by_op: dict[str, list[tuple[float, float]]] = {}
    for plane in xspace.planes:
        if not plane.name.startswith("/device:GPU:"):
            continue
        stat_meta_name = {m.id: m.name for m in plane.stat_metadata.values()}
        event_meta_stats = {
            m.id: list(m.stats) for m in plane.event_metadata.values()
        }

        def _resolve_hlo_op(stats) -> Optional[str]:
            for stat in stats:
                if stat_meta_name.get(stat.metadata_id) != "hlo_op":
                    continue
                which = stat.WhichOneof("value")
                if which == "str_value":
                    return stat.str_value
                if which == "ref_value":
                    return stat_meta_name.get(stat.ref_value)
            return None

        for line in plane.lines:
            for event in line.events:
                if event.WhichOneof("data") != "offset_ps":
                    continue
                dur_us = event.duration_ps / 1e6
                if dur_us <= 0:
                    continue
                hlo_op = (
                    _resolve_hlo_op(event_meta_stats.get(event.metadata_id, ()))
                    or _resolve_hlo_op(event.stats)
                )
                if not hlo_op or not hlo_op.startswith(_TE_EP_PREFIX):
                    continue
                ts_us = (line.timestamp_ns * 1000 + event.offset_ps) / 1e6
                by_op.setdefault(hlo_op, []).append((ts_us, dur_us))

    return _average_invocation_costs(by_op)


def _resolve_reference_trace_path(spec: str) -> Optional[str]:
    """Resolve COLLECTIVE_OVERLAP_REFERENCE_TRACE to an actual trace file.

    The trace's own path is never stable across runs -- it's nested under
    per-run output dirs keyed by job ID/timestamp and a per-host subdir keyed
    by hostname/capture-timestamp (e.g.
    <output_dir>/<output_dir>/tensorboard/plugins/profile/<ts>/<host>.trace.json.gz).
    Requiring an exact path would mean updating this env var by hand every
    run. Instead, `spec` may be:
      - an exact file: used as-is.
      - a directory: searched recursively for *.trace.json.gz.
      - a glob pattern (containing * or ?): expanded directly, e.g.
        ".../outputs/*/*/tensorboard/plugins/profile/*/*.trace.json.gz".
    Whenever more than one file matches, the most recently modified one wins
    -- so pointing this at a stable parent directory (or even the whole
    shared outputs/ root) and just leaving it there works across runs
    without any manual upkeep, as long as it's updated by copying/symlinking
    a known-good run's trace there once you have one you trust.
    """
    import glob  # pylint: disable=import-outside-toplevel

    if os.path.isfile(spec):
        return spec
    pattern = spec if any(c in spec for c in "*?[") else os.path.join(spec, "**", "*.trace.json.gz")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        _logger.warning(
            "collective_overlap_pass: COLLECTIVE_OVERLAP_REFERENCE_TRACE=%s "
            "matched no trace files.", spec,
        )
        return None
    return max(matches, key=os.path.getmtime)


def _apply_te_ep_cost_overrides() -> None:
    """Override PGLE's te_ep_* entries in _profile_costs with corrected
    per-invocation costs loaded once from COLLECTIVE_OVERLAP_REFERENCE_TRACE.

    If that env var isn't set, falls back to searching $WORKSPACE_DIR/outputs
    -- the maxtext-launcher submit script always exports WORKSPACE_DIR
    (submit.template.sh: --export=ALL,WORKSPACE_DIR=$WORKSPACE_DIR), and
    every run's trace lands somewhere under
    $WORKSPACE_DIR/outputs/<run>/<run>/tensorboard/plugins/profile/<ts>/, so
    this works out of the box with zero config -- it just picks up whatever
    the most recently modified trace anywhere in outputs/ happens to be. Set
    COLLECTIVE_OVERLAP_REFERENCE_TRACE explicitly to pin a specific, curated
    trace instead of trusting "most recent" (e.g. if the latest run in
    outputs/ was a broken or otherwise unrepresentative one).

    No-op (and cheap to call repeatedly) once loaded or if neither is set.
    """
    global _te_ep_overrides_loaded
    if _te_ep_overrides_loaded:
        if _te_ep_overrides:
            _profile_costs.update(_te_ep_overrides)
        return
    _te_ep_overrides_loaded = True
    spec = os.environ.get("COLLECTIVE_OVERLAP_REFERENCE_TRACE")
    if not spec:
        workspace_dir = os.environ.get("WORKSPACE_DIR")
        if not workspace_dir:
            return
        spec = os.path.join(workspace_dir, "outputs")
    trace_path = _resolve_reference_trace_path(spec)
    if trace_path is None:
        return
    _te_ep_overrides.update(_load_te_ep_costs_from_trace(trace_path))
    if _te_ep_overrides:
        _profile_costs.update(_te_ep_overrides)
        _logger.info(
            "collective_overlap_pass: loaded %d te_ep_* cost corrections from "
            "reference trace %s (resolved from %s).",
            len(_te_ep_overrides), trace_path, spec,
        )
    else:
        _logger.warning(
            "collective_overlap_pass: reference trace %s (resolved from %s) "
            "had no te_ep_* entries.", trace_path, spec,
        )


# ---------------------------------------------------------------------------
# Cross-rank sync (JAX's own distributed coordination service)
# ---------------------------------------------------------------------------
# Each rank's PGLE profile differs slightly (hardware/timing noise), and
# since the pass's decisions are a deterministic function of
# _profile_costs, different ranks could schedule the same SPMD program
# differently -- illegal, since all ranks must compile an identical
# executable.
#
# Tried and rejected: broadcasting rank 0's _profile_costs (syncing the
# *input*) -- insufficient, ranks still diverged even with matching inputs.
# An external cross-process broadcast library for the *output* bytes
# instead worked, but needed its process group to span all ranks, which
# needed extra Slurm launch flags that interfered with NCCL's own
# cross-node bootstrap and caused a different hang.
#
# What we use instead: JAX's own distributed coordination service
# (jax._src.distributed.global_state.client, already connected on every
# multi-process job) as a plain key-value store -- mirrors how stock
# JAX/XLA's AutoPGLE syncs FDO profiles
# (jax/_src/compiler.py:_share_fdo_profiles). One process publishes bytes
# under a content-derived key; everyone else does a blocking read on that
# key. No extra library, no extra Slurm flags, no NCCL bootstrap collision.
# We publish the pass's *output* bytes here (output-broadcast is the
# strategy that actually eliminates divergence, per above).
_dist_client = None
_dist_checked = False

# Monotonic count of _collective_overlap_pass invocations that proceeded
# past the early-out (i.e. had an interesting async op to potentially
# reorder). Used to name the entry barrier below -- see its call site for
# why this needs to be call-order-based rather than content-hash-based.
_barrier_invocation_count = 0


def _get_distributed_client():
    """Lazily resolve jax's distributed coordination-service client.

    Logs the resolved process_id/process_count at INFO on every process so a
    mis-launched job (e.g. jax.distributed.initialize() never called, or
    called with num_processes=1) is immediately visible in the logs rather
    than failing silently: a process_count=1 "world" makes every rank act as
    its own root, which defeats the whole point of sharing rank 0's result
    with everyone else.
    """
    global _dist_client, _dist_checked
    if not _dist_checked:
        _dist_checked = True
        try:
            from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel
            _dist_client = _jax_distributed.global_state.client
            _process_id = _jax_distributed.global_state.process_id
            _process_count = _jax_distributed.global_state.num_processes
            _logger.info(
                "collective_overlap_pass: JAX distributed client resolved: "
                "process_id=%d process_count=%d (host=%s, pid=%d).%s",
                _process_id, _process_count, socket.gethostname(), os.getpid(),
                "" if (_dist_client is not None and _process_count > 1) else (
                    " WARNING: client=%s process_count=%d means this process "
                    "cannot share its scheduled module with other ranks -- "
                    "every rank will compute its own (potentially divergent) "
                    "schedule. This usually means jax.distributed.initialize() "
                    "was never called, or was called with num_processes=1."
                    % (_dist_client is not None, _process_count)
                ),
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: jax distributed client unavailable "
                "(%s); cannot share rank 0's scheduled module, so each rank "
                "will compute (and use) its own locally-scheduled module. "
                "This risks ranks ending up with divergent schedules.", exc,
            )
            _dist_client = None
    return _dist_client


def _jax_process_id() -> int:
    from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel
    return _jax_distributed.global_state.process_id


def _jax_process_count() -> int:
    from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel
    return _jax_distributed.global_state.num_processes


# Timeout for the key-value share below. Mirrors JAX's own default for the
# analogous FDO-profile share (jax_share_binary_between_hosts_timeout_ms).
_SHARE_TIMEOUT_MS = int(
    os.environ.get("COLLECTIVE_OVERLAP_SHARE_TIMEOUT_MS", str(20 * 60 * 1000))
)

# Sentinel prefix bytes distinguishing "no transformation" (None) from actual
# module bytes in the key-value store, since the value there must be bytes.
_SHARE_NONE = b"\x00"
_SHARE_SOME = b"\x01"


# ---------------------------------------------------------------------------
# POST_SCHEDULER pass — data structures
# ---------------------------------------------------------------------------
# Dedicated "Start" opcodes (kebab-case, via _opcode_str below) for async
# collectives that have their own Start/Done opcode pair. Some collectives
# (reduce-scatter, notably -- it has no dedicated Start/Done opcode pair at
# all) are instead represented via the generic kAsyncStart/kAsyncDone
# wrapper; unlike the dedicated opcodes, a generic "async-start" doesn't
# need its wrapped op inspected to know it's worth scheduling -- it's async
# by construction, so it's checked separately in _is_async_start below.
_ASYNC_START_OPCODES = frozenset({
    "all-gather-start",
    "all-reduce-start",
    "collective-permute-start",
})

# Minimum deficit (µs) to consider a collective for phase-2 splitting.
# Override with COLLECTIVE_OVERLAP_DISABLE_SPLIT=1 to run baseline (no split).
_SPLIT_DEFICIT_THRESHOLD_US = (
    float("inf") if os.environ.get("COLLECTIVE_OVERLAP_DISABLE_SPLIT") == "1"
    else 500.0
)

# Set COLLECTIVE_OVERLAP_WHILE_BODY_ONLY=1 to skip phase-1 reorder/fill (and
# therefore phase-2 split, which only ever runs on collectives phase-1
# actually queued as split candidates) entirely for the module's true entry
# computation, processing only while-loop body computations. The entry
# computation (e.g. `main`) is by far the largest sequence in a full-scale
# module and dominates compile time -- the candidate scan in
# _fill_exposed_collectives_with_heavy_compute is O(len(seq) - done_pos) per
# while-loop iteration with no bound on window width, and deep chase chains
# (dozens of hops seen in practice on `main`, e.g. job 3204127's 20-minute
# timeout) multiply that by however many hops fire. While-loop bodies are
# orders of magnitude shorter, so this knob trades away entry-computation
# overlap entirely for a much faster compile when iterating.
_WHILE_BODY_ONLY = os.environ.get("COLLECTIVE_OVERLAP_WHILE_BODY_ONLY") == "1"
# Set COLLECTIVE_OVERLAP_HOIST_FSDP_STARTS=0 to disable the final per-while-body
# step that moves every FSDP all-gather/reduce-scatter start (plus its trivial
# operand chain, after hoisting any control-predecessors pinning it) to
# its earliest legal position regardless of deficit.
_HOIST_FSDP_STARTS = os.environ.get("COLLECTIVE_OVERLAP_HOIST_FSDP_STARTS", "1") != "0"
# Set COLLECTIVE_OVERLAP_HOIST_ASYNC_DONE_GUARD=1 to refuse hoisting an async-done
# predecessor when that would shrink its own start's window below its profiled
# latency. Off by default: in job 3212382 this guard refused fusion-done.16/.4
# (small async stash fusions) 22 times, which kept every FSDP collective after
# them pinned late.
_HOIST_ASYNC_DONE_GUARD = os.environ.get("COLLECTIVE_OVERLAP_HOIST_ASYNC_DONE_GUARD", "0") == "1"
# Set COLLECTIVE_OVERLAP_FSDP_TE_EP_AWARE=0 to disable te_ep-aware placement of FSDP
# starts in the hoist step (see _fsdp_te_ep_defer_target).
_FSDP_TE_EP_AWARE = os.environ.get("COLLECTIVE_OVERLAP_FSDP_TE_EP_AWARE", "0") == "1"
# Set COLLECTIVE_OVERLAP_FSDP_STREAM_MODEL=0 to disable the communication-stream
# model used to place FSDP starts/dones in the hoist step (see
# _best_fsdp_placement). When on, it supersedes the te_ep-aware heuristic above.
_FSDP_STREAM_MODEL = os.environ.get("COLLECTIVE_OVERLAP_FSDP_STREAM_MODEL", "1") != "0"
# Instruction-name prefixes _phase2_split_core actually knows how to split:
# a plain batched collective custom-call with multiple tuple operands
# directly on the async-start. Other async-start-wrapped ops (e.g.
# call-start wrapping a te_ep collective) don't have a directly-splittable
# operand list in the shape _phase2_split_core expects -- it just logs "no
# inner collective" and produces nothing for them, so there's no point
# queuing them as split candidates in the first place.
#
# Matched against ag_start.name (not .opcode): all-gather/reduce-scatter/
# all-reduce-start are lowered as a generic async-start wrapper at the HLO
# opcode level (inst.opcode.name comes back as "async-start" for all of
# them, same as a te_ep call-start) -- only the instruction's *name* (its
# scheduling_name, e.g. "all-gather-start.8") actually distinguishes which
# kind of collective it is. Same pattern as _FSDP_COLLECTIVE_PREFIXES below.
_SPLITTABLE_NAME_PREFIXES = ("all-gather-start", "reduce-scatter-start", "all-reduce-start")

# Minimum number of operand groups to split.
_SPLIT_MIN_GROUPS = 2
# Typical number of operands per "layer epoch" in a batched collective.
# 34 operands / 5 FFN layers ≈ 7 → 5 groups.  Used for rank-based grouping
# when there are no natural position gaps between layers.
_SPLIT_GROUP_SIZE = 7


@dataclass
class _SplitCandidate:
    start_name: str
    done_name: str
    deficit_us: float
    comp_name: str = ""
    # effective_producer_pos[i] = schedule position of the 'real' producer
    # (tracing through bitcasts/GTEs) for operand i of the async-start.
    effective_producer_pos: list[int] = field(default_factory=list)
    # PGLE-measured latency of the *original*, pre-split collective. Used by
    # _phase2_split_core to estimate each new sub-collective's own latency
    # (proportional to its share of total operand bytes) -- the new
    # sub-collectives have no PGLE profile entry of their own (they didn't
    # exist when profiling ran), so without this they'd be silently skipped
    # by the post-split _phase1_reorder re-run (see _run_phase1_to_fixed_point).
    total_latency_us: float = 0.0


def _is_async_start(inst: object) -> bool:
    """True if `inst` is the Start half of an async collective we care about.

    Checked by opcode (via _opcode_str), not by instruction name -- names
    are compiler-assigned labels (from user annotations, dedup, etc.), not
    a reliable signal of what an instruction actually is; only the opcode
    is authoritative. all-gather/all-reduce/collective-permute each have a
    dedicated kXStart opcode for their async Start half
    (_ASYNC_START_OPCODES). reduce-scatter has no dedicated Start/Done
    opcode pair -- XLA represents it via the generic kAsyncStart/kAsyncDone
    wrapper instead. A generic kAsyncStart doesn't need its wrapped op
    inspected: being async at all is sufficient to know phase 1 should
    consider scheduling it, so any "async-start" opcode counts directly.

    Note: a plain collective-shaped instruction (e.g. a while-loop body's
    ROOT all-gather) with collective_backend_config.is_sync=false is NOT a
    separate case to handle here -- it's just the *inner* representation of
    a normal async-start/async-done pair (the plain op is the ROOT of the
    async_wrapped_computation an outer "async-start ... calls=%async_comp"
    instruction calls). The outer async-start is what appears in the
    schedule this function walks, and it's already covered by the
    "async-start" opcode check above.

    A generic "async-start" opcode is not exclusively a network collective:
    with use_generic_async_start_done, XLA routes essentially everything
    async through this one opcode -- standard NCCL collectives (whether
    printed as "all-gather-start"/"reduce-scatter-start"/etc. with direct
    operands and a collective_backend_config, or as generic "async-start
    ... calls=%async_computation.N"), TE's expert-parallel dispatch/combine
    ops (wrapped via to_apply=%name, tagged
    frontend_attributes={_xla_stream_annotation="collective"} -- see
    _resolve_profile_key), and even
    apparently-unrelated async "call" wrappers with a trivial passthrough
    body that carry no meaningful cost. All of these genuinely share the
    same opcode value, not just similar printed keywords (verified: a
    reduce-scatter-start printed with direct operands and no calls= still
    satisfies this check, and previously an attempt to require calls=/the
    stream annotation here to exclude the trivial-passthrough case instead
    silently broke detection for that direct-operand collective form --
    there is no cheap, reliable way to distinguish "trivial passthrough
    call" from "genuine collective" by opcode/attributes alone, so this
    deliberately stays permissive: being async at all is enough to know
    phase 1 should consider scheduling it, and a trivial-cost wrapper is
    harmless here since it will simply never show a deficit worth acting
    on).
    """
    opcode = _opcode_str(inst)
    return opcode in _ASYNC_START_OPCODES or opcode == "async-start"


def _module_has_interesting_async_ops(module) -> bool:
    """True if any non-fusion computation in the module has an async-start op.

    Checked against every non-fusion computation (not just the entry one) --
    an async-start can live inside a while-body/condition computation just
    as well as the entry computation, and make_nonfusion_computations() is
    the same accessor _innermost_first_computations() uses, so this sees
    exactly the set of computations phase 1 would otherwise walk. Fusion
    bodies are excluded since collectives never appear inside a fusion.

    A cheap, single pass over instruction opcodes -- used as an early-out
    before touching the distributed client/barrier at all, so the (much more
    common) modules with no collectives to reorder skip all synchronization
    overhead entirely.
    """
    for comp in module.make_nonfusion_computations():
        for inst in comp.instructions():
            if _is_async_start(inst):
                return True
    return False


def _opcode_str(inst) -> str:
    """Canonical hyphenated opcode string (e.g. "get-tuple-element") for a
    phase-1 C++ HloInstruction.

    inst.opcode returns a jaxlib._hlo.HloOpcode enum (e.g. HloOpcode.kFusion),
    not a plain string, so comparing it directly against string literals or
    string sets is always False.  This converts the enum's CamelCase member
    name (minus the leading 'k') to XLA's kebab-case opcode spelling.
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "-", inst.opcode.name[1:]).lower()


_HEAVY_INPUT_BYTES = 8 * 1024 * 1024  # 8 MB threshold for fusion inputs

# XLA PrimitiveType → bytes per element (xla_data.proto enum values)
_PTYPE_BYTES: dict[int, int] = {
    1: 1,                           # PRED
    2: 1, 3: 2, 4: 4, 5: 8,        # S8/S16/S32/S64
    6: 1, 7: 2, 8: 4, 9: 8,        # U8/U16/U32/U64
    10: 2, 11: 4, 12: 8, 16: 2,    # F16/F32/F64/BF16
    15: 8, 18: 16,                  # C64/C128
    19: 1, 20: 1,                   # F8E5M2/F8E4M3FN
}


def _proto_shape_bytes(shape) -> int:
    """Total bytes for an HloShapeProto (tuples summed recursively)."""
    if shape.element_type == 13:  # TUPLE
        return sum(_proto_shape_bytes(s) for s in shape.tuple_shapes)
    n = 1
    for d in shape.dimensions:
        n *= d
    return n * _PTYPE_BYTES.get(shape.element_type, 4)


def _is_heavy_kernel_proto(inst, id_to_inst: dict) -> bool:
    """True for cublas/cudnn custom-calls or large fusion ops (≥8 MB input).

    These are SM-saturating kernels that can delay NIC startup for collectives
    scheduled immediately after them.
    """
    if inst.opcode == "custom-call":
        tgt = getattr(inst, "custom_call_target", "").lower()
        return "cublas" in tgt or "cudnn" in tgt
    if inst.opcode == "fusion":
        total = sum(
            _proto_shape_bytes(id_to_inst[oid].shape)
            for oid in inst.operand_ids
            if oid in id_to_inst
        )
        return total >= _HEAVY_INPUT_BYTES
    return False


def _effective_producer_pos(inst, positions: dict, max_depth: int = 8) -> int:
    """Return the schedule position of inst's 'real' producer, looking through
    zero-cost ops (bitcast, get-tuple-element, tuple) up to max_depth steps."""
    _ZERO_COST = ("bitcast", "get-tuple-element", "tuple")
    cur = inst
    for _ in range(max_depth):
        if _opcode_str(cur) not in _ZERO_COST:
            break
        ops = list(cur.operands())
        if not ops or ops[0] not in positions:
            break
        cur = ops[0]
    return positions.get(cur, positions.get(inst, 0))


def _proto_effective_pos(
    op_id: int,
    id_to_inst: dict,
    id_to_sched_pos: dict,
    max_depth: int = 8,
) -> int:
    """Like _effective_producer_pos but works on proto instruction objects by ID.

    Traces through zero-cost ops (bitcast, get-tuple-element, tuple) to find
    the 'real' producer's schedule position in the CURRENT proto schedule.
    Used by the split subprocess to avoid relying on stale phase-1 positions.
    """
    _ZERO_COST = ("bitcast", "get-tuple-element", "tuple")
    cur_id = op_id
    for _ in range(max_depth):
        inst = id_to_inst.get(cur_id)
        if inst is None or inst.opcode not in _ZERO_COST:
            break
        if not inst.operand_ids:
            break
        next_id = inst.operand_ids[0]
        if next_id not in id_to_sched_pos:
            break
        cur_id = next_id
    pos = id_to_sched_pos.get(cur_id)
    if pos is None:
        pos = id_to_sched_pos.get(op_id, 0)
    return pos


# ---------------------------------------------------------------------------
# Trivially-movable operand helpers (phase 1 operand relocation)
# ---------------------------------------------------------------------------
_TRIVIAL_SINGLE_OPCODES = frozenset({
    "bitcast", "convert", "transpose", "reshape", "broadcast",
    "get-tuple-element", "tuple", "copy", "slice",
    # "parameter"/"constant" are leaves with no real data dependency (a
    # parameter is available from the start of the computation; a constant
    # is compile-time), so they should always be freely movable -- but they
    # were missing here even though _TRIVIAL_FUSED_OPCODES already treats
    # them as trivial inside a fusion body. Without this, the chain walk
    # treated wherever a param/constant happened to already sit as a
    # genuine blocker, and the 0us-cost gate (meant to skip cheap *fill
    # candidates*) then refused to relocate it either. Confirmed via job
    # 3174209: this was the single largest refusal reason blocking
    # all-gather-start.8.g3's chase from ever reaching te_gemm_v2_ffi.81/.87
    # or dot_product_attention_fwd.10.
    "parameter", "constant",
})

_TRIVIAL_FUSED_OPCODES = frozenset({
    "parameter", "constant", "iota",
    "convert", "bitcast", "reshape", "transpose", "broadcast", "copy",
    "concatenate", "dynamic-slice", "slice",
    "get-tuple-element", "tuple",
    "add", "subtract", "multiply", "divide", "negate", "abs",
    "maximum", "minimum", "and", "or", "not", "xor",
    "sqrt", "rsqrt", "exp", "log", "sign", "clamp",
    "floor", "ceil", "round-nearest-afz", "round-nearest-even",
    "compare", "select",
})


_CALLS_RE = re.compile(r"calls=%([A-Za-z0-9_.]+)")


def _is_trivial_fusion_body(inst, comp_by_name: dict) -> bool:
    """True if every op inside inst's called (fused) computation is trivial.

    The phase-1 HloInstruction binding (jax._src.lib.hlo) has no
    called_computations()/instructions() accessor, so the callee computation
    name is recovered from inst.to_string() (which always prints
    "calls=%name" for a fusion) and looked up in a module-wide name->comp map.
    """
    try:
        text = inst.to_string()
    except Exception:
        return False
    m = _CALLS_RE.search(text)
    if not m:
        return False
    comp = comp_by_name.get(m.group(1))
    if comp is None:
        return False
    try:
        for fi in comp.instructions():
            if _opcode_str(fi) not in _TRIVIAL_FUSED_OPCODES:
                return False
        return True
    except Exception:
        return False


def _is_trivially_movable_inst(inst, comp_by_name: dict) -> bool:
    """True if inst can be safely relocated in the schedule without heavy compute cost."""
    opc = _opcode_str(inst)
    if opc in _TRIVIAL_SINGLE_OPCODES:
        return True
    if opc == "fusion":
        return _is_trivial_fusion_body(inst, comp_by_name)
    return False


_CONTROL_PRED_RE = re.compile(r"control-predecessors=\{([^}]*)\}")


def _control_predecessor_names(inst) -> list[str]:
    """Names of inst's control-predecessors.

    The jaxlib HloInstruction Python binding used in phase 1 (jax._src.lib.hlo)
    does not expose control_predecessors() directly, so we recover them from
    the instruction's textual form, which always prints them when present.
    """
    try:
        text = inst.to_string()
    except Exception:
        return []
    m = _CONTROL_PRED_RE.search(text)
    if not m:
        return []
    return [n.strip().lstrip("%") for n in m.group(1).split(",") if n.strip()]


_TO_APPLY_RE = re.compile(r"to_apply=%([A-Za-z0-9_.]+)")
_ROOT_RE = re.compile(r"\bROOT %([A-Za-z0-9_.\-]+) = ")


def _computation_root(comp):
    """Find comp's ROOT instruction, via comp.root_instruction().

    Falls back to parsing comp.to_string()'s "ROOT %name = ..." marker (the
    same signal XLA's own printer uses) and then to "the one instruction
    with no in-computation users" only if root_instruction() itself isn't
    available for some reason -- that last heuristic is unreliable in
    general (an unused parameter also has no users, and can be mispicked
    instead of the real root) but better than nothing.
    """
    try:
        return comp.root_instruction()
    except Exception:
        pass
    try:
        text = comp.to_string()
        m = _ROOT_RE.search(text)
        if m:
            root_name = m.group(1)
            for inst in comp.instructions():
                if inst.name == root_name:
                    return inst
    except Exception:
        pass
    try:
        for inst in comp.instructions():
            if not list(inst.users()):
                return inst
    except Exception:
        pass
    return None


_MAX_PROFILE_KEY_UNWRAP_HOPS = 8


def _unwrap_to_leaf_name(inst, comp_by_name: dict) -> str:
    """Follow nested call/async wrappers down to the instruction PGLE
    actually profiles.

    Per the observed structure of TE's expert-parallel dispatch/combine
    wrapping: an async-start whose own computation's root is a plain "call"
    (kCall, synchronous, not async) needs one more hop through *that* call's
    own to_apply= computation -- its root is the profiled instruction. A
    get-tuple-element (or any other non-call/async op) reached at any point
    is treated as a legitimate stopping point, not something to unwrap
    further: it may simply have no profile entry (a trivial passthrough,
    same as we already tolerate for standard collectives), which is fine --
    we don't force our way past it. Bounded by _MAX_PROFILE_KEY_UNWRAP_HOPS
    purely as a safety net against unexpectedly deep or cyclic call chains.
    """
    for _hop in range(_MAX_PROFILE_KEY_UNWRAP_HOPS):
        opc = _opcode_str(inst)
        if opc not in ("call", "call-start", "async-start"):
            return inst.name
        try:
            text = inst.to_string()
        except Exception:
            return inst.name
        m = _TO_APPLY_RE.search(text) or _CALLS_RE.search(text)
        if not m:
            return inst.name
        comp = comp_by_name.get(m.group(1))
        if comp is None:
            return inst.name
        root = _computation_root(comp)
        if root is None:
            return inst.name
        inst = root
    return inst.name


def _resolve_profile_key(ag_start, comp_by_name: dict) -> str:
    """Best-effort name to look up ag_start's measured latency under in
    _profile_costs.

    Standard collectives (all-gather-start etc, opcode async-start wrapping
    calls=%async_computation.N) expose their wrapped root directly via
    async_wrapped_root(), and PGLE keys their cost under that root
    instruction's name.

    TE's expert-parallel dispatch/combine ops are wrapped differently:
    opcode async-start too, but via to_apply=%name (a plain HLO call, not
    the calls=%async_computation.N convention) and tagged with
    frontend_attributes={_xla_stream_annotation="collective"} (XLA's own
    marker for "scheduled on the collective stream"). async_wrapped_root()
    doesn't understand the to_apply= convention and raises for those, so we
    fall back to parsing to_apply=%name ourselves and finding that
    computation's ROOT via _computation_root.

    Either way, the root found (whether via async_wrapped_root() or the
    to_apply= fallback) is NOT assumed to already be the final leaf: it is
    always passed through _unwrap_to_leaf_name, which additionally chases
    it further if it's itself another call/async wrapper rather than the
    real kernel. This matters because async_wrapped_root() can genuinely
    succeed yet still return an intermediate wrapper (verified empirically:
    for te_ep's call-start instructions it reliably resolved to a plain
    "call.N" instruction, one hop short of the real te_ep_dispatch_ffi.N/
    te_ep_combine_ffi.N leaf) -- returning on its first success without
    unwrapping further silently skipped all of the fallback logic below.
    """
    root = None
    try:
        root = ag_start.async_wrapped_root()
    except Exception:
        root = None
    if root is None:
        try:
            text = ag_start.to_string()
        except Exception:
            text = ""
        m = _TO_APPLY_RE.search(text)
        if m:
            comp = comp_by_name.get(m.group(1))
            if comp is not None:
                root = _computation_root(comp)
    if root is None:
        return ag_start.name
    return _unwrap_to_leaf_name(root, comp_by_name)


def _is_te_ep_call_start(inst, comp_by_name: dict) -> bool:
    """True if inst is an async-start (call-start) whose resolved profile
    key (see _resolve_profile_key) is a te_ep_combine_ffi/te_ep_dispatch_ffi
    leaf -- i.e. one of the big MoE expert-parallel dispatch/combine
    windows (multi-ms latency, see the region_20 call-start.54/56/59/64/67/
    69 investigation earlier this session)."""
    if not _is_async_start(inst):
        return False
    return _resolve_profile_key(inst, comp_by_name).startswith(("te_ep_combine", "te_ep_dispatch"))


def _relocate_fsdp_done_before_te_ep(
    ag_start, ag_done, comp, schedule, seq: list, positions: dict,
    name_to_pos: dict, comp_by_name: dict, collective_latency: float,
    module_name: str = "",
):
    """For an FSDP collective that's already fully hidden (real overlap >=
    its own latency), pull its `done` instruction forward to the earliest
    point that's still fully hidden AND immediately before a te_ep
    dispatch/combine call-start, if one exists in that range.

    `done`'s program-order position is what legally gates every instruction
    that consumes the collective's result (a bitcast/GTE reading
    all-gather-done.N's output, and transitively anything downstream of
    that) -- it can sit anywhere strictly after `start` and still be
    correct, completely independent of how much real slack the window has
    beyond what's needed to hide the latency. Left wherever phase-1's own
    (start-only) relocation happens to leave it, `done` often ends up with
    a lot of unnecessary slack after the point the collective is actually
    hidden -- during which none of its real downstream consumers are even
    legally schedulable yet, even though the real data transfer is already
    done in every way that matters. Moving `done` up to right where the
    latency is hidden (and no further -- going earlier than that would
    newly expose the collective, the opposite of the goal) frees those
    consumers to become fill candidates for whatever's scheduled between
    the new and old `done` position -- specifically targeting the point
    just before a te_ep call-start, since those are the largest, hardest-
    to-fill windows in this model (multi-ms each) and most starved for
    legally-reachable heavy-compute candidates.

    Always legal regardless of how far forward `done` moves (down to the
    latency-closure point): `done`'s only real operand is `start` itself,
    so moving it anywhere after `start` can never violate a data
    dependency; its own consumers already sit after its *old* position by
    construction, so they stay after its new (earlier) position too.
    Control-predecessors (rare for an async-done) are still respected via
    `_control_predecessor_names`.

    Returns (changed, seq, positions, name_to_pos).
    """
    start_pos = positions[ag_start]
    done_pos = positions[ag_done]
    if done_pos <= start_pos + 1:
        return False, seq, positions, name_to_pos

    done_floor = start_pos + 1
    for name in _control_predecessor_names(ag_done):
        cp_pos = name_to_pos.get(name)
        if cp_pos is not None and cp_pos + 1 > done_floor:
            done_floor = cp_pos + 1

    cum = 0.0
    latency_closure_pos = None
    for i in range(start_pos + 1, done_pos):
        cum += _resolve_inst_cost(seq[i], comp_by_name)
        if cum >= collective_latency:
            latency_closure_pos = i + 1
            break
    if latency_closure_pos is None:
        return False, seq, positions, name_to_pos

    search_start = max(latency_closure_pos, done_floor)
    target = None
    for i in range(search_start, done_pos):
        inst = seq[i]
        if inst is ag_done or inst is ag_start:
            continue
        if _is_te_ep_call_start(inst, comp_by_name):
            target = i
            break
    if target is None:
        return False, seq, positions, name_to_pos

    new_seq = [inst for inst in seq if inst is not ag_done]
    new_seq.insert(target, ag_done)
    schedule.set_sequence(comp, new_seq)
    new_positions = {inst: i for i, inst in enumerate(new_seq)}
    new_name_to_pos = {inst.name: i for inst, i in new_positions.items()}
    _logger.info(
        "collective_overlap_pass [%s]: moved %s done from pos %d to %d, "
        "right before te_ep call-start %s -- still fully hidden (latency "
        "%.1f us closed by pos %d), frees downstream consumers to help "
        "fill te_ep windows.",
        module_name, ag_done.name, done_pos, target, seq[target].name,
        collective_latency, latency_closure_pos,
    )
    return True, new_seq, new_positions, new_name_to_pos


_MAX_RELOCATE_CHAIN = 64

# Minimum profiled cost (us) for a non-trivial instruction to be considered
# a "heavy compute" worth relocating into an exposed collective's window --
# below this it's not worth the bookkeeping/relocation churn.
_HEAVY_COMPUTE_MIN_US = 5.0

_TE_GEMM_TARGET_RE = re.compile(r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*)"')


def _is_te_gemm_custom_call(inst, comp_by_name: dict) -> bool:
    """True if inst is a custom-call to a MoE GEMM kernel (custom_call_target
    prefixed te_grouped_gemm or te_gemm, e.g. te_grouped_gemm_v2_ffi.N or
    te_gemm_v2_ffi.N), OR a kind=kCustom fusion (e.g. a dynamic-slice-fusion)
    whose single nested computation directly wraps one -- the same wrapper
    case _resolve_inst_cost corrects for when summing costs, since a
    kCustom fusion's own name never gets a PGLE entry (the real kernel
    launch is tagged with the inner custom-call's name instead). Without
    this, a te_gemm wrapped in a dynamic-slice-fusion would get the right
    cost (via _resolve_inst_cost) but never win priority here, since its
    own opcode is "fusion", not "custom-call".

    These dominate compute time in DeepSeek-family MoE models, so when
    filling an exposed FSDP-style all-gather/reduce-scatter collective's
    window (see _fill_exposed_collectives_with_heavy_compute), they're
    prioritized over other legal heavy-compute candidates.
    """
    opc = _opcode_str(inst)
    try:
        text = inst.to_string()
    except Exception:
        return False
    if opc == "custom-call":
        return bool(_TE_GEMM_TARGET_RE.search(text))
    if opc == "fusion" and "kind=kCustom" in text:
        m = _CALLS_RE.search(text)
        if not m:
            return False
        comp = comp_by_name.get(m.group(1))
        if comp is None:
            return False
        try:
            for sub in comp.instructions():
                if _opcode_str(sub) != "custom-call":
                    continue
                if _TE_GEMM_TARGET_RE.search(sub.to_string()):
                    return True
        except Exception:
            return False
        return False
    return False


_FSDP_COLLECTIVE_PREFIXES = ("all-gather", "reduce-scatter")

_HEAVY_ANCHOR_TARGET_RE = re.compile(
    r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*|[^"]*cudnn[^"]*|[^"]*cublas[^"]*)"'
)


def _is_heavy_anchor_custom_call(inst, comp_by_name: dict) -> bool:
    """True if inst is a custom-call (or a kind=kCustom fusion wrapping one,
    same unwrap as _is_te_gemm_custom_call) to te_grouped_gemm/te_gemm/cudnn/
    cublas -- the specific set of kernels _chase_ag_start_blocker_toward_
    heavy_compute treats as a "good enough, stop chasing" landing anchor for
    an FSDP collective's own blocker chain. Broader than
    _is_te_gemm_custom_call (which only prioritizes te_gemm/te_grouped_gemm
    for fill-candidate selection) -- kept as a separate function/regex so
    this new, less-validated chase path doesn't silently change the
    existing, already-validated FSDP fill-priority behavior.
    """
    opc = _opcode_str(inst)
    try:
        text = inst.to_string()
    except Exception:
        return False
    if opc == "custom-call":
        return bool(_HEAVY_ANCHOR_TARGET_RE.search(text))
    if opc == "fusion" and "kind=kCustom" in text:
        m = _CALLS_RE.search(text)
        if not m:
            return False
        comp = comp_by_name.get(m.group(1))
        if comp is None:
            return False
        try:
            for sub in comp.instructions():
                if _opcode_str(sub) != "custom-call":
                    continue
                if _HEAVY_ANCHOR_TARGET_RE.search(sub.to_string()):
                    return True
        except Exception:
            return False
        return False
    return False


def _earliest_legal_pos(
    ag_start,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    cp_pins: Optional[list] = None,
) -> tuple[int, list, object]:
    """Compute the earliest position ag_start can legally be relocated to.

    If cp_pins is a list, it is filled with (position, name) for every
    control-predecessor outside the moving set (callers use it to find which
    one pins the floor).

    Walks ag_start's operands transitively through trivially-movable
    instructions (bitcast/reshape/elementwise ops, trivial fusions, etc.) —
    unlike a fixed relocation window, this walk has no distance bound, since
    a trivial/zero-cost op is free to move arbitrarily far back.  The walk
    stops at any non-trivial ("real compute") instruction, which pins a data-
    dependency floor: the move can be no earlier than right after it.  Every
    control-predecessor of a moved instruction (or of ag_start itself) that
    is not itself part of the moving set pins a control-dependency floor the
    same way.

    Returns (floor, movable, blocker) where floor is the smallest legal
    insertion position (in the current, pre-move schedule) and movable is
    the subset of the trivial chain that actually needs to relocate
    alongside ag_start (sorted in schedule order, excluding ag_start itself)
    -- i.e. those chain members currently positioned at or after floor.
    Chain members already positioned before floor satisfy the ordering
    constraint as-is and are left untouched; forcing them to move too would
    needlessly drag ag_start's own final position later (floor +
    len(movable)), potentially past its original position, negating the
    point of the move. blocker is the single non-trivial *data*-dependency
    instruction whose position actually pins floor (None if floor is 0, or
    if what pins it is a control-predecessor rather than a data producer) --
    callers can try relocating a movable blocker earlier in its own right to
    see if that opens up more room.
    """
    candidates: set = set()
    seen: set = set()
    floor = 0
    blocker = None
    queue = list(ag_start.operands())
    while queue:
        inst = queue.pop()
        if inst in seen:
            continue
        seen.add(inst)
        if inst not in positions:
            continue
        if len(candidates) < _MAX_RELOCATE_CHAIN and _is_trivially_movable_inst(inst, comp_by_name):
            candidates.add(inst)
            queue.extend(inst.operands())
        else:
            if positions[inst] + 1 > floor:
                floor = positions[inst] + 1
                blocker = inst

    moving_names = {inst.name for inst in candidates} | {ag_start.name}
    for inst in list(candidates) + [ag_start]:
        for name in _control_predecessor_names(inst):
            if name in moving_names:
                continue
            cp_pos = name_to_pos.get(name)
            if cp_pos is not None and cp_pins is not None:
                cp_pins.append((cp_pos, name))
            if cp_pos is not None and cp_pos + 1 > floor:
                floor = cp_pos + 1
                blocker = None  # a control-predecessor, not a relocatable data producer

    # floor only grows monotonically as the walk visits more of the operand
    # tree, so a candidate discovered early on may already sit before the
    # *final* floor -- only relocate the ones that don't.
    movable = [inst for inst in candidates if positions[inst] >= floor]
    return floor, sorted(movable, key=lambda i: positions[i]), blocker


# Bound on how many times, per collective, we chase "the blocker's own
# blocker" earlier (e.g. a GEMM blocked by an even earlier GEMM). The chain
# is meant to be walked in full, all the way back to the computation's true
# inputs (parameters) or another collective's done -- each hop's own
# _earliest_legal_pos call already has no distance bound, and the loop
# self-terminates the moment a blocker has no legal room left (or is too
# cheap to bother relocating). This cap only guards against a pathological
# or buggy chain that never terminates; it should not normally be reached.
_MAX_PRODUCER_RELOCATE_HOPS = 64

# A recursive variant of the direction-(a) chain-relocation fix below (chase
# a collective's own blocker's own blocker, and so on) was tried and removed
# -- measured net-negative (job 3150632: 36.25% overlap vs. 37.85% baseline
# and 39.34% for the simpler non-recursive fix). Believed cause:
# _find_best_blocker_position's net-benefit check is a per-hop
# approximation that doesn't account for a later hop undoing an earlier
# hop's "net win" assumption, so individually-positive moves can still sum
# to a net-negative schedule. See git history if this ever gets revisited.

# Bound on how many times _fill_exposed_collectives_with_heavy_compute will
# chase "the blocker's own blocker" for a single fill candidate that isn't
# yet legally reachable. Unlike the recursive collective's-own-blocker
# chase above (removed, net-negative), this chases a *fill candidate's*
# blocker via _try_relocate_blocker_earlier, whose net-benefit gate already
# re-checks total exposed time before committing each hop -- individually
# gated, though the same cross-hop risk could in principle still apply. Set
# COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS=0 to disable.
_MAX_HEAVY_COMPUTE_CHASE_HOPS = int(
    os.environ.get("COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS", "8")
)

# _fill_exposed_collectives_with_heavy_compute used to build its `windows`
# worklist once, as a snapshot of whichever collectives were exposed at
# entry -- so a collective fully hidden at that instant never got added,
# and if a *later* window's chase eroded its coverage, nothing would notice
# or refill it until the next (costly) schedule.update()-mediated fixed
# point in _run_phase1_to_fixed_point. This cap bounds an outer loop (the
# `for _fp_iter in range(...)` below) that rebuilds the worklist from every
# collective's fresh deficit each pass and keeps going until a pass makes
# no further changes, so newly-exposed windows get picked back up
# immediately instead of waiting for that costly outer fixed point.
_MAX_FILL_FIXED_POINT_ITERS = 3

# A chase that makes partial-but-insufficient progress on a candidate
# causes the scan to restart from done_pos+1, by design (a deep chain often
# needs several such restarts). But if the same candidate is the first
# blocked thing found on every restart, this can consume the whole
# chase_hops budget on one never-resolving candidate while other,
# genuinely reachable candidates never get a turn. Give up on a candidate
# once it's triggered this restart cycle this many times in a row without
# resolving, so the scan can move past it.
_STUCK_CANDIDATE_GIVE_UP_STREAK = 4


def _resolve_inst_cost(inst, comp_by_name: dict) -> float:
    """Profiled cost (us) for inst, correcting for kind=kCustom fusions
    (e.g. a dynamic-slice-fusion) whose PGLE cost is attributed to the real
    instruction(s) inside their single nested computation rather than to
    the outer fusion wrapper.

    Example: a dynamic-slice-fusion wrapping a real kernel like
    te_gemm_v2_ffi.N alongside a dynamic-update-slice that writes its
    result into a loop-carried buffer -- PGLE tags each inner GPU kernel
    launch with its own hlo_op, so _profile_costs has entries under
    "te_gemm_v2_ffi.N" (and "dynamic-update-slice.N" too, if it has a
    measurable cost of its own) but never under the fusion's own name, so a
    plain _profile_costs.get(inst.name, 0.0) silently sees 0 for the whole
    fusion.

    Unlike the call/call-start chain (_unwrap_to_leaf_name), a kCustom
    fusion's body can hold MULTIPLE real, independently-costed ops rather
    than a single terminal leaf -- its ROOT is often just a trivial `tuple`
    combining them (e.g. ROOT %tuple.81 = tuple(%dynamic-update-slice.145,
    %te_gemm_v2_ffi.354#1)) -- so the correct fix is to SUM every
    instruction in the body that has its own profile entry, not to resolve
    down to "the" leaf.

    Falls back to the plain inst.name lookup (0.0 if absent) whenever inst
    isn't a kCustom fusion, or its body sums to 0 anyway.
    """
    direct = _profile_costs.get(inst.name)
    if direct:
        return direct
    if _opcode_str(inst) != "fusion":
        return 0.0
    try:
        text = inst.to_string()
    except Exception:
        return 0.0
    if "kind=kCustom" not in text:
        return 0.0
    m = _CALLS_RE.search(text)
    if not m:
        return 0.0
    comp = comp_by_name.get(m.group(1))
    if comp is None:
        return 0.0
    try:
        return sum(_profile_costs.get(sub.name, 0.0) for sub in comp.instructions())
    except Exception:
        return 0.0


def _prefix_costs_excluding(seq: list, exclude: set, comp_by_name: dict) -> list:
    """Prefix sum of profile costs along `seq`, with instructions in
    `exclude` contributing 0 -- lets overlap for a hypothetical position be
    computed without those instructions' cost being double-counted at both
    their old and a candidate new position."""
    prefix = [0.0] * (len(seq) + 1)
    for i, inst in enumerate(seq):
        c = 0.0 if inst in exclude else _resolve_inst_cost(inst, comp_by_name)
        prefix[i + 1] = prefix[i] + c
    return prefix


def _total_exposed_us(
    start_of_done: dict, positions: dict, seq: list, comp_by_name: dict,
) -> float:
    """Sum of positive deficits (exposed/un-hidden latency) across every
    collective in `start_of_done` with a known profile cost, using the
    schedule exactly as it stands -- the ground-truth "how much is exposed
    right now, everywhere" metric used to decide whether a candidate
    relocation is a net win or a net loss."""
    prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    total = 0.0
    for ag_done, ag_start in start_of_done.items():
        if ag_start not in positions or ag_done not in positions:
            continue
        profile_key = _resolve_profile_key(ag_start, comp_by_name)
        latency = _profile_costs.get(profile_key)
        if latency is None or latency <= 0:
            continue
        s, d = positions[ag_start], positions[ag_done]
        overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
        total += max(0.0, latency - overlap)
    return total


def _find_best_blocker_position(
    blocker,
    chain: list,
    floor: int,
    blocker_pos: int,
    ag_start,
    ag_done_pos: int,
    ag_latency: float,
    start_of_done: dict,
    positions: dict,
    seq: list,
    comp_by_name: dict,
) -> tuple[int, float]:
    """Search positions in [floor, blocker_pos] for the one that minimizes
    TOTAL exposed time summed across every collective in this computation --
    not just the one `blocker` is currently pinning -- rather than always
    relocating all the way to the theoretical-earliest floor.

    Relocating blocker to position P affects two kinds of window:
      - ag_start's own window: once blocker (and its trivial chain) lands
        at P, ag_start's own new floor becomes P + len(chain) + 1, so its
        overlap is recomputed directly against that hypothetical position.
      - every *other* collective's window [start2, done2): if blocker
        currently sits inside it but wouldn't at P, that window LOSES
        blocker's cost from its overlap; if the reverse, it GAINS blocker's
        cost. Windows containing both, neither, or unaffected by the move
        see no change. (Chain members are themselves trivial -- bitcast /
        reshape / GTE / elementwise -- so they carry ~0 profiled cost and
        are ignored for this windowing accounting; only blocker's own
        placement matters.)

    This is an approximation (pre-move positions are used to decide window
    containment for candidates, and chain members straddling a window
    boundary aren't split out individually), but it is cheap -- O(number of
    collectives) per candidate position -- and catches the dominant effect:
    relocating a heavy op out of a stretch of the schedule that other,
    already-hidden collectives were relying on for their own overlap.

    Returns (best_pos, best_total_exposed_us) where best_pos is the chain's
    insertion start -- blocker itself lands at best_pos + len(chain), same
    convention as _earliest_legal_pos/_phase1_reorder use elsewhere.
    best_pos == blocker_pos - len(chain) means no candidate landing position
    actually reduces total exposed time, i.e. don't move blocker at all.
    """
    prefix = _prefix_costs_excluding(seq, set(chain) | {blocker}, comp_by_name)
    block_cost = _resolve_inst_cost(blocker, comp_by_name)

    other_windows = []  # (start_pos, done_pos, latency, base_overlap_excl_block)
    for od, os in start_of_done.items():
        if os is ag_start or os not in positions or od not in positions:
            continue
        profile_key = _resolve_profile_key(os, comp_by_name)
        latency2 = _profile_costs.get(profile_key)
        if latency2 is None or latency2 <= 0:
            continue
        s2, d2 = positions[os], positions[od]
        base_overlap = prefix[d2] - prefix[s2 + 1] if d2 > s2 + 1 else 0.0
        other_windows.append((s2, d2, latency2, base_overlap))

    # Search directly over blocker's *landing* position, not the chain's
    # insertion start -- that way "leave blocker where it is" is
    # unambiguously new_pos == blocker_pos, with no implicit dependence on
    # len(chain). (Evaluating candidates in insertion-start units instead,
    # as an earlier version of this function did, made the "no-op" baseline
    # silently equal to blocker_pos + len(chain) -- a real, later position
    # -- any time the chain was non-empty, which corrupted the net-benefit
    # comparison against the true current state and could make relocating
    # back and forth between two blockers each look like a net win in
    # sequence, i.e. a non-terminating oscillation.) The insertion-start P
    # actually used to splice the schedule is derived from the winning
    # landing position by subtracting len(chain) exactly once, at the end.
    lo = floor + len(chain)
    new_pos_candidates = {lo, blocker_pos}
    for s2, d2, _, _ in other_windows:
        if lo <= s2 < blocker_pos:
            new_pos_candidates.add(s2 + 1)
        if lo < d2 <= blocker_pos:
            new_pos_candidates.add(d2)
    new_pos_candidates = sorted(p for p in new_pos_candidates if lo <= p <= blocker_pos)

    def total_exposed(blocker_new_pos: int) -> float:
        ag_new_start = blocker_new_pos + 1
        ag_overlap = (
            prefix[ag_done_pos] - prefix[ag_new_start] if ag_done_pos > ag_new_start else 0.0
        )
        total = max(0.0, ag_latency - ag_overlap)
        for s2, d2, latency2, base_overlap in other_windows:
            contains_new = s2 < blocker_new_pos < d2
            overlap2 = base_overlap + (block_cost if contains_new else 0.0)
            total += max(0.0, latency2 - overlap2)
        return total

    best_new_pos, best_exposed = blocker_pos, None
    for new_pos in new_pos_candidates:
        e = total_exposed(new_pos)
        # Prefer strictly lower total exposed time; on a tie, prefer the
        # position closest to blocker_pos (the least disruptive change that
        # achieves the same result).
        if best_exposed is None or e < best_exposed - 1e-6 or (
            abs(e - best_exposed) <= 1e-6 and new_pos > best_new_pos
        ):
            best_exposed = e
            best_new_pos = new_pos
    return best_new_pos - len(chain), best_exposed


def _try_relocate_blocker_earlier(
    blocker,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    ag_start,
    ag_done_pos: int,
    ag_latency: float,
    start_of_done: dict,
    module_name: str = "",
    position_margins: dict | None = None,
):
    """If `blocker` -- a non-trivial instruction pinning some collective's
    floor -- is itself a movable heavy op, relocate it (and its own trivial
    operand chain) earlier, via one of two acceptance paths:

    1. Whichever legal position between its current spot and its own
       theoretical-earliest floor minimizes TOTAL exposed time across every
       collective in this computation (see _find_best_blocker_position) --
       not necessarily all the way to that floor. This is the original,
       strict path: a hop is only accepted if it visibly improves the
       aggregate metric.

    2. (Only if `position_margins` is given -- i.e. only from the
       heavy-compute chase, never from the disabled general recursive
       chase, to avoid resurrecting whatever made that net-negative) a
       margin-safe fallback straight to `floor`: accepted if blocker's own
       cost never exceeds the slack (_compute_position_margins) of
       whatever collective(s) currently cover its position, even if this
       specific hop doesn't show up as an aggregate improvement. This
       exists because path 1's net-exposure gate is myopic across a chain:
       relocating a small op by one or two positions often looks like a
       pure wash in isolation (its own contribution to any window's
       overlap sum is negligible either way), even when it's a necessary
       link in a longer chain that only pays off once the ORIGINAL
       candidate several hops downstream finally becomes reachable -- see
       the all-gather-start.8.g3 / input_reduce_fusion.60 investigation
       (job 3179392), where input_reduce_fusion.60 had genuine room to
       move but every hop was refused by path 1 despite being perfectly
       margin-safe. Path 2 is weaker than path 1 (no aggregate-improvement
       requirement) but still strictly safe: it can never push a
       currently-fine collective's overlap below its own latency.

    Moving a producer to an earlier position can never break its own
    downstream consumers: in any valid schedule a consumer already sits
    after its producer, so it still does after the producer moves to an
    even earlier (but still legal) position. What CAN happen is that pulling
    it out of its current spot steals overlap headroom some *other*
    already-hidden collective was relying on -- which is exactly what both
    paths above guard against, just via two different safety criteria.

    Returns the refreshed (seq, positions, name_to_pos) if a move happened,
    else None.
    """
    if blocker not in positions:
        _logger.debug(
            "collective_overlap_pass [%s]: TRC %s: blocker not in positions "
            "(already relocated/removed this round?), refusing.",
            module_name, blocker.name,
        )
        return None
    cost = _resolve_inst_cost(blocker, comp_by_name)
    if cost < _HEAVY_COMPUTE_MIN_US:
        _logger.debug(
            "collective_overlap_pass [%s]: TRC %s: cost=%.1f us < "
            "_HEAVY_COMPUTE_MIN_US=%.1f us, refusing to chase.",
            module_name, blocker.name, cost, _HEAVY_COMPUTE_MIN_US,
        )
        return None
    floor, chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
    blocker_pos = positions[blocker]
    # blocker's own trivial chain gets inserted immediately before it, so
    # its actual landing position is floor + len(chain), not floor itself.
    # Comparing blocker_pos against the bare floor here would treat a
    # blocker that's already perfectly packed right after its chain (no
    # gap at all) as having "room to move", trigger a move that reinserts
    # everything at the exact same final positions, and report success --
    # which, since nothing about blocker_pos would actually change, makes
    # every subsequent call see the same false "room to move" again,
    # forever (bounded only by _MAX_PRODUCER_RELOCATE_HOPS).
    final_pos = floor + len(chain)
    if final_pos >= blocker_pos:
        _logger.debug(
            "collective_overlap_pass [%s]: TRC %s: no room to move "
            "(own floor=%d + chain=%d = %d >= current pos=%d).",
            module_name, blocker.name, floor, len(chain), final_pos, blocker_pos,
        )
        return None  # blocker itself already has nowhere earlier to go

    best_pos, best_exposed = _find_best_blocker_position(
        blocker, chain, floor, blocker_pos, ag_start, ag_done_pos, ag_latency,
        start_of_done, positions, seq, comp_by_name,
    )
    # best_pos is an insertion-start position; blocker_pos - len(chain) is
    # the insertion-start that reproduces "leave blocker exactly where it
    # is" (blocker_pos itself is a landing position, not an insertion
    # start -- see _find_best_blocker_position's docstring).
    current_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
    improves_total = (
        best_pos < blocker_pos - len(chain) and best_exposed < current_total - 1e-6
    )

    target = None
    if improves_total:
        target = best_pos
        _logger.debug(
            "collective_overlap_pass [%s]: TRC %s: relocating from pos %d "
            "to %d (chain len %d) -- total exposed %.1f -> %.1f us "
            "(path 1: aggregate improvement).",
            module_name, blocker.name, blocker_pos, best_pos + len(chain),
            len(chain), current_total, best_exposed,
        )
    elif position_margins is not None:
        margin = position_margins.get(blocker_pos, float("inf"))
        if cost <= margin:
            target = floor
            _logger.debug(
                "collective_overlap_pass [%s]: TRC %s: relocating from pos "
                "%d to %d (chain len %d) -- no aggregate improvement "
                "(best_exposed=%.1f us vs current=%.1f us) but margin-safe "
                "(cost=%.1f us <= margin=%.1f us at its own position) "
                "(path 2: margin-safe).",
                module_name, blocker.name, blocker_pos, floor + len(chain),
                len(chain), best_exposed, current_total, cost, margin,
            )
        else:
            _logger.debug(
                "collective_overlap_pass [%s]: TRC %s: no aggregate "
                "improvement (best_exposed=%.1f us vs current=%.1f us) AND "
                "not margin-safe (cost=%.1f us > margin=%.1f us at its own "
                "position) -- refusing.",
                module_name, blocker.name, best_exposed, current_total,
                cost, margin,
            )
    else:
        _logger.debug(
            "collective_overlap_pass [%s]: TRC %s: own floor=%d is legal "
            "(pos %d, room to move to %d), but no candidate landing "
            "position in [floor, current] reduces total exposed time "
            "(best_exposed=%.1f us vs current=%.1f us) -- refusing.",
            module_name, blocker.name, floor, blocker_pos, final_pos,
            best_exposed, current_total,
        )

    if target is None:
        return None

    to_move_set = set(chain) | {blocker}
    new_seq = [inst for inst in seq if inst not in to_move_set]
    ins_pos = target
    for inst in chain:  # already in topological (schedule) order
        new_seq.insert(ins_pos, inst)
        ins_pos += 1
    new_seq.insert(ins_pos, blocker)
    schedule.set_sequence(comp, new_seq)
    new_positions = {inst: i for i, inst in enumerate(new_seq)}
    new_name_to_pos = {inst.name: i for inst, i in new_positions.items()}
    return new_seq, new_positions, new_name_to_pos


def _chase_ag_start_blocker_toward_heavy_compute(
    ag_start,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    max_hops: int,
    module_name: str = "",
):
    """FSDP-only: when ag_start itself has no legal earlier position because
    its own direct (non-trivial) data producer -- the `blocker` from
    _earliest_legal_pos -- sits right next to it, relocate that blocker
    earlier too (unconditionally, same remove+reinsert move the rest of
    this function's caller uses for ag_start itself -- no net-exposure or
    margin check), repeating until either the blocker chain reaches a
    recognized heavy-compute anchor (te_grouped_gemm/te_gemm/cudnn/cublas,
    see _is_heavy_anchor_custom_call) or no further room exists.

    _earliest_legal_pos only walks ag_start's *trivial* operand chain and
    stops cold at the first non-trivial producer -- it never asks whether
    that producer itself has room to move. So "cannot move" can be
    misleading: an unrelated, genuinely heavy instruction (e.g. a GEMM)
    with zero data dependency on ag_start can sit even further back in the
    schedule, structurally reachable, while ag_start stays stuck because
    nothing ever tried moving its own producer out of the way. Confirmed in
    practice: all-gather-start.8.g0 in region_7.19_spmd.clone.1 (job
    3204236) reported "cannot move" with its direct producers
    (te_dbias_quantize_ffi.810/.825/.828/.819) pinning the floor, while
    te_grouped_gemm_ffi.96.double_buffer_clone -- no data dependency on the
    all-gather at all -- sat comfortably earlier in the same computation.

    No safety gate (unlike _try_relocate_blocker_earlier, used by the other
    chase functions in this file): an earlier version of this routed
    through that function's net-exposure/margin-safe checks, but those
    exist to stop a relocation from stealing overlap from some *other*
    collective's window that currently depends on the relocated
    instruction being where it is. For FSDP all-gather/reduce-scatter
    collectives specifically, that risk is low in practice -- their direct
    operand producers (quantize/dbias ops feeding the collective) are
    narrow and specific to that one collective, not generally shared with
    other collectives' overlap windows the way a general heavy-compute
    candidate might be. Dropped deliberately for simplicity/speed; if
    validation ever shows this regressing some other collective's
    exposure, that's the signal the assumption doesn't hold and the gate
    needs to come back.

    Returns (changed, seq, positions, name_to_pos, floor, to_move, blocker)
    -- floor/to_move/blocker are ag_start's current _earliest_legal_pos
    result after however many hops fired, exactly like
    _chase_heavy_compute_blocker_chain's return shape, so the caller can
    immediately proceed with its own move-to-floor logic.
    """
    changed = False
    floor, to_move, blocker = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
    hops = 0
    while (
        blocker is not None
        and not _is_heavy_anchor_custom_call(blocker, comp_by_name)
        and hops < max_hops
    ):
        b_floor, b_chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
        blocker_pos = positions[blocker]
        final_pos = b_floor + len(b_chain)
        if final_pos >= blocker_pos:
            _logger.debug(
                "collective_overlap_pass [%s]: CAB %s: blocker %s has no "
                "room itself (own floor=%d + chain=%d = %d >= current "
                "pos=%d) -- stopping chain.",
                module_name, ag_start.name, blocker.name, b_floor,
                len(b_chain), final_pos, blocker_pos,
            )
            break
        to_move_set = set(b_chain) | {blocker}
        new_seq = [inst for inst in seq if inst not in to_move_set]
        ins_pos = b_floor
        for inst in b_chain:  # already in topological (schedule) order
            new_seq.insert(ins_pos, inst)
            ins_pos += 1
        new_seq.insert(ins_pos, blocker)
        schedule.set_sequence(comp, new_seq)
        seq = new_seq
        positions = {inst: i for i, inst in enumerate(seq)}
        name_to_pos = {inst.name: i for inst, i in positions.items()}
        changed = True
        hops += 1
        prev_blocker = blocker
        floor, to_move, blocker = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
        _logger.debug(
            "collective_overlap_pass [%s]: CAB %s: hop %d relocated %s "
            "from pos %d to %d -- new floor=%d, next blocker=%s.",
            module_name, ag_start.name, hops, prev_blocker.name, blocker_pos,
            final_pos, floor, blocker.name if blocker is not None else None,
        )
    return changed, seq, positions, name_to_pos, floor, to_move, blocker


def _chase_exposed_collective_blocker_earlier(
    blocker_done,
    start_of_done: dict,
    exposed_dones: set,
    window_start_pos: int,
    window_done_pos: int,
    seq: list,
    schedule,
    comp,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
):
    """If `blocker_done` is the done-side of ANOTHER still-exposed collective
    (its own window still has unhidden deficit -- see `exposed_dones`), try
    to relocate that collective's own start earlier so it sits between the
    caller's window and whatever heavy-compute candidate it's currently
    blocking, instead of leaving that candidate permanently disqualified.

    Unlike _try_relocate_blocker_earlier (which chases an arbitrary heavy op
    blocking a collective, and was found net-negative when applied
    recursively to a collective's own blocker -- see the removed
    recursive-chase code, job 3150632), this only fires for the narrow,
    concrete case the caller has already identified: a real data dependency
    on another exposed collective's done sitting between this window and a
    candidate that would otherwise fill it. Relocating that collective's
    start closer to the caller's window can only help both: it gets a
    chance at overlap from the caller's window, and the candidate's own
    floor (pinned by that done) moves earlier too.

    The move is capped at `window_done_pos` -- it never lands past the end
    of the caller's own window -- and bounded by the chased collective's own
    legal floor, so it never violates its own dependencies.

    Returns (changed, seq, positions, name_to_pos); the seq/positions/
    name_to_pos are the original objects, unchanged, if no move was made.
    """
    if blocker_done not in exposed_dones:
        return False, seq, positions, name_to_pos
    chase_start = start_of_done.get(blocker_done)
    if chase_start is None or chase_start not in positions:
        return False, seq, positions, name_to_pos
    chase_start_pos = positions[chase_start]
    if chase_start_pos <= window_start_pos:
        # Already at or before our window -- nothing to gain by chasing it.
        return False, seq, positions, name_to_pos
    chase_floor, chase_chain, _ = _earliest_legal_pos(
        chase_start, positions, name_to_pos, comp_by_name
    )
    target = max(chase_floor, window_start_pos + 1)
    target = min(target, window_done_pos)
    if target + len(chase_chain) >= chase_start_pos:
        # No real room to move it earlier than where it already sits.
        return False, seq, positions, name_to_pos

    to_move_set = set(chase_chain) | {chase_start}
    new_seq = [inst for inst in seq if inst not in to_move_set]
    ins_pos = target
    for inst in chase_chain:  # already in topological (schedule) order
        new_seq.insert(ins_pos, inst)
        ins_pos += 1
    new_seq.insert(ins_pos, chase_start)
    schedule.set_sequence(comp, new_seq)
    new_positions = {inst: i for i, inst in enumerate(new_seq)}
    new_name_to_pos = {inst.name: i for inst, i in new_positions.items()}
    return True, new_seq, new_positions, new_name_to_pos


def _find_relocatable_ancestor(
    inst, positions: dict, name_to_pos: dict, comp_by_name: dict, max_depth: int,
    skip_cheap_ancestors: bool = False,
):
    """Walk `inst`'s own blocker chain until finding an ancestor with
    genuine *structural* room to move -- its own floor + trivial chain
    lands strictly before its current position -- as opposed to one that's
    already exactly as early as its own dependencies allow.

    By default, purely structural: does not consider cost or net exposure
    at all, so the returned instruction (if any) still MUST be run through
    _try_relocate_blocker_earlier's full gating before actually being
    moved. That gate -- unchanged, exactly as it already protects a
    single-hop chase -- is what refuses to disrupt an already-hidden
    collective; this function only ever decides *which* instruction to
    offer to that gate, never bypasses it.

    skip_cheap_ancestors additionally requires cost >= _HEAVY_COMPUTE_MIN_US
    before accepting an ancestor, skipping past trivial near-zero-cost
    "connector" ops (dynamic-slice fusions, select fusions, a collective's
    own call-start/done) that have structural room but would just get
    refused by _try_relocate_blocker_earlier's own cost gate anyway --
    without skipping them, the chase gives up on the whole chain the
    instant it hits one, even with hop budget left and a genuinely heavy,
    reachable candidate sitting further back (confirmed job 3202509).

    Enabling this unconditionally measured net-negative overall (job
    3203932 vs. 3202899, same config): 132 vs. 79 successful relocations
    (confirming it does find more candidates) but the net total exposed
    time still got WORSE (83724.4us vs. 72879.4us) -- almost entirely
    concentrated in `main`, the computation with the deepest/longest chase
    chains, even though a smaller while-loop computation (region_19)
    clearly improved (11513.3us -> 5742.5us). Same root cause as why
    `_ENABLE_RECURSIVE_BLOCKER_CHASE` is disabled (job 3150632):
    `_find_best_blocker_position`'s net-benefit check is a per-hop
    approximation that doesn't account for a later hop undoing an earlier
    hop's "net win" assumption, so individually-positive moves can still
    sum to a net-negative schedule -- more hops firing (exactly what this
    flag causes) means more chances for that compounding error to
    accumulate, and `main`'s much longer chase chains give it far more
    opportunity to compound than a short while-loop body. So this is
    scoped to while-body computations only (see `is_while_body` threaded
    down from `_phase1_reorder`), where the same change measured a clear
    net win in isolation.

    Returns the relocatable ancestor (possibly `inst` itself), or None if
    the chain bottoms out (no further blocker, a dependency cycle, or
    max_depth exhausted) before finding one.
    """
    seen = set()
    cur = inst
    for _ in range(max_depth + 1):
        if cur is None or cur in seen or cur not in positions:
            return None
        seen.add(cur)
        floor, chain, sub_blocker = _earliest_legal_pos(cur, positions, name_to_pos, comp_by_name)
        has_room = floor + len(chain) < positions[cur]
        if has_room and (
            not skip_cheap_ancestors
            or _resolve_inst_cost(cur, comp_by_name) >= _HEAVY_COMPUTE_MIN_US
        ):
            return cur
        cur = sub_blocker
    return None


def _chase_heavy_compute_blocker_chain(
    candidate,
    comp,
    schedule,
    seq: list,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    ag_start,
    ag_done_pos: int,
    ag_latency: float,
    start_of_done: dict,
    max_hops: int,
    module_name: str = "",
    position_margins: dict | None = None,
    is_while_body: bool = False,
):
    """Repeatedly try to relocate whatever's currently blocking `candidate`
    from reaching a legal position inside window [ag_start, ag_done_pos),
    up to `max_hops` times.

    `position_margins` (see _compute_position_margins), if given, is
    forwarded to _try_relocate_blocker_earlier to enable its margin-safe
    fallback acceptance path -- without it, a hop is only accepted if it
    visibly improves aggregate total exposed time, which is often too
    myopic to ever unblock a candidate several hops away (see that
    function's docstring, path 2).

    Complements _chase_exposed_collective_blocker_earlier (which only
    chases a blocker that's specifically another exposed collective's
    done): here the blocker can be *any* movable heavy op -- e.g.
    dot_product_attention_fwd blocked by a GEMM that's itself blocked by
    another GEMM two hops further back, the case that motivated this (see
    the all-gather-start.8.g3 investigation: te_gemm_v2_ffi.81/.87 and the
    attention op consuming them were all structurally reachable but none of
    them were ever chased into place, because the fill scan only asks "is
    the candidate legal *right now*", never "would relocating its blocker
    first make it legal").

    A blocker that has no room to move on its own (already sitting exactly
    at its own floor) doesn't end the chain anymore: before giving up, each
    hop first walks *structurally* down that blocker's own blocker chain
    (_find_relocatable_ancestor, no mutation, no gating) to find an
    ancestor that genuinely can move, then hands *that* instruction to
    _try_relocate_blocker_earlier -- unchanged, still the sole thing that
    actually commits a move, and still only ever commits one if it strictly
    reduces _total_exposed_us across every collective in the computation.
    So a hop that would help `candidate` but hurt an already-hidden
    collective elsewhere is refused exactly as before; this only changes
    *which* instruction gets offered to that gate, never bypasses it. Two
    hops are typically needed to actually unblock `candidate` through an
    indirect blocker: one to relocate the deep ancestor, a second (the next
    iteration of this same while loop, since `candidate`'s own blocker is
    re-derived fresh after every hop) to relocate the now-newly-movable
    direct blocker itself.

    Stops as soon as a hop is refused or no relocatable ancestor exists
    within the remaining hop budget (no further hops attempted from
    there), reaches max_hops, or resolves the candidate's own floor.

    Returns (changed, seq, positions, name_to_pos, floor, chain, blocker)
    -- floor/chain/blocker are candidate's *current* _earliest_legal_pos
    result after however many hops fired, so the caller can immediately
    recheck `floor <= ag_done_pos` without a redundant extra call.
    """
    changed = False
    floor, chain, blocker = _earliest_legal_pos(candidate, positions, name_to_pos, comp_by_name)
    _logger.debug(
        "collective_overlap_pass [%s]: CHC %s: initial floor=%d (window "
        "done_pos=%d, gap=%d), blocker=%s, max_hops=%d.",
        module_name, candidate.name, floor, ag_done_pos, floor - ag_done_pos,
        blocker.name if blocker is not None else None, max_hops,
    )
    hops = 0
    while floor > ag_done_pos and blocker is not None and hops < max_hops:
        target = _find_relocatable_ancestor(
            blocker, positions, name_to_pos, comp_by_name, max_hops - hops,
            skip_cheap_ancestors=is_while_body,
        )
        if target is None:
            _logger.debug(
                "collective_overlap_pass [%s]: CHC %s: blocker %s (and its "
                "own blocker chain) has no relocatable ancestor within the "
                "remaining hop budget (%d left) -- stopping chain.",
                module_name, candidate.name, blocker.name, max_hops - hops,
            )
            break
        if target is not blocker:
            _logger.debug(
                "collective_overlap_pass [%s]: CHC %s: blocker %s has no "
                "room itself -- descending to its own blocker %s instead.",
                module_name, candidate.name, blocker.name, target.name,
            )
        result = _try_relocate_blocker_earlier(
            target, comp, schedule, seq, positions, name_to_pos, comp_by_name,
            ag_start, ag_done_pos, ag_latency, start_of_done, module_name,
            position_margins,
        )
        hops += 1
        if result is None:
            _logger.debug(
                "collective_overlap_pass [%s]: CHC %s: hop %d refused to "
                "relocate %s (see preceding TRC line for reason) -- "
                "stopping chain (floor still %d > done_pos %d).",
                module_name, candidate.name, hops, target.name,
                floor, ag_done_pos,
            )
            break
        seq, positions, name_to_pos = result
        changed = True
        if position_margins is not None:
            # Recompute fresh after every hop, not just once per outer
            # while-iteration: a single chase call can commit up to
            # max_hops relocations back-to-back, and reusing the
            # pre-chase margin snapshot across all of them would let
            # several individually-safe-looking hops cumulatively draw
            # more from the same window's slack than it actually has --
            # each hop must see the *current* remaining slack, not the
            # slack as of before this chase call started.
            position_margins = _compute_position_margins(
                start_of_done, positions, seq, comp_by_name
            )
        prev_floor = floor
        floor, chain, blocker = _earliest_legal_pos(
            candidate, positions, name_to_pos, comp_by_name
        )
        _logger.debug(
            "collective_overlap_pass [%s]: CHC %s: hop %d relocated %s -- "
            "floor %d -> %d (done_pos=%d), next blocker=%s.",
            module_name, candidate.name, hops, target.name, prev_floor, floor,
            ag_done_pos, blocker.name if blocker is not None else None,
        )
    if floor > ag_done_pos:
        _logger.debug(
            "collective_overlap_pass [%s]: CHC %s: gave up after %d/%d "
            "hop(s), still floor=%d > done_pos=%d (gap=%d)%s.",
            module_name, candidate.name, hops, max_hops, floor, ag_done_pos,
            floor - ag_done_pos,
            " -- ran out of hops" if hops >= max_hops
            else " -- blocker exhausted or last hop refused",
        )
    else:
        _logger.debug(
            "collective_overlap_pass [%s]: CHC %s: resolved after %d "
            "hop(s) -- floor=%d <= done_pos=%d.",
            module_name, candidate.name, hops, floor, ag_done_pos,
        )
    return changed, seq, positions, name_to_pos, floor, chain, blocker


def _compute_position_margins(
    start_of_done: dict, positions: dict, seq: list, comp_by_name: dict,
    prefix: list | None = None, min_pos: int = 0,
) -> dict:
    """For every schedule position >= min_pos, the minimum slack (current
    overlap minus latency) among all collectives whose [start_pos+1,
    done_pos) window covers it right now -- i.e. how much cost could be
    pulled out of that position without dropping ANY covering collective's
    overlap below its own latency. A position covered by no collective
    simply has no entry (callers treat that as infinite margin via
    .get(i, inf)).

    min_pos lets a caller that only ever queries positions >= some bound
    (e.g. a candidate scan starting at done_pos+1) skip both collectives
    whose window ends before that bound entirely and the portion of a
    straddling window's range below it -- this is the module-wide,
    O(sum of all window widths) part of the fill loop's per-iteration cost,
    called fresh on every single candidate placement, so trimming it to
    only the range that will actually be queried matters a lot. Safe only
    when the caller's own query range is itself bounded below by min_pos;
    a caller that queries arbitrary earlier positions (e.g. TRC's
    margin-safe fallback, which checks a blocker's *current*, possibly far
    earlier, position) must keep the default min_pos=0.

    Generalizes what used to be a binary covered/not-covered distinction
    (a position was either fully off-limits or fully free): a position
    sitting inside a massively over-covered window -- e.g. one with 30x
    more overlap than its latency needs -- is just as safe to relocate from
    as genuinely idle time, as long as no more than the real slack is
    taken. Refusing ANY move touching a covered position regardless of
    margin was needlessly conservative. Confirmed via the DIAG WATCH
    instrumentation on job 3179259: te_gemm_v2_ffi.81/.87/.78/.84 and
    dot_product_attention_fwd.10 were permanently excluded from
    all-gather-start.8.g3's window because they sit inside windows like
    all-gather-start.11's (overlap=36885us against a latency of just
    2690us -- ~34ms of pure slack) even though stealing a few hundred
    microseconds of GEMM cost from there could never have exposed it.

    Used to restrict heavy-compute fill candidates so a direct placement
    (below) can never silently push some other collective's overlap below
    its own latency -- the blocker-chase path already gets the equivalent
    protection from _find_best_blocker_position's whole-computation
    net-exposure check; this is that same guarantee for direct placement.
    """
    if prefix is None:
        prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    margins: dict[int, float] = {}
    for ag_done, ag_start in start_of_done.items():
        if ag_start not in positions or ag_done not in positions:
            continue
        profile_key = _resolve_profile_key(ag_start, comp_by_name)
        latency = _profile_costs.get(profile_key)
        if latency is None or latency <= 0:
            continue
        s, d = positions[ag_start], positions[ag_done]
        if d <= s + 1 or d <= min_pos:
            continue
        overlap = prefix[d] - prefix[s + 1]
        slack = overlap - latency
        for p in range(max(s + 1, min_pos), d):
            if p not in margins or slack < margins[p]:
                margins[p] = slack
    return margins


_GIVE_UP_MODES = ("none", "streak", "floor_aware")


class _StuckCandidateTracker:
    """Decides when a repeatedly-encountered blocked candidate should be
    given up on (skipped in the scan) so other candidates get a turn --
    see the caller for why this matters. Three strategies, tried against
    each other per computation by _fill_exposed_collectives_best_of (see
    that function's docstring for the measured trade-off between them):

    - "none": never gives up -- the original always-retry behavior.
    - "streak": tracks only the single most-recently-stuck candidate name
      and a running count; a *different* name interrupting the streak
      resets it, even for the original candidate. Cheap and, empirically,
      the safest general default -- see job 3184713.
    - "floor_aware": tracks every candidate independently and only counts
      a repeat if its _earliest_legal_pos floor failed to improve since
      last seen (genuine multi-hop progress, however many hops it takes,
      never counts against it). More precise in principle, but empirically
      let some truly-stuck candidates (whose floor inches down slightly
      without ever actually resolving) consume more budget than "streak"
      does -- see job 3187886.
    """

    def __init__(self, give_up_mode: str):
        self.give_up_mode = give_up_mode
        self._last_name: str | None = None
        self._last_streak = 0
        self._streak_by_name: dict[str, int] = {}
        self._last_floor_by_name: dict[str, int] = {}

    def note_and_maybe_give_up(self, name: str, floor: int) -> tuple[int, bool]:
        """Record that `name` was just found blocked at `floor`. Returns
        (streak, should_give_up_now)."""
        if self.give_up_mode == "none":
            return 0, False
        if self.give_up_mode == "streak":
            if name == self._last_name:
                self._last_streak += 1
            else:
                self._last_name = name
                self._last_streak = 1
            streak = self._last_streak
        elif self.give_up_mode == "floor_aware":
            prev_floor = self._last_floor_by_name.get(name)
            if prev_floor is not None and floor >= prev_floor:
                self._streak_by_name[name] = self._streak_by_name.get(name, 0) + 1
            else:
                self._streak_by_name[name] = 1
            self._last_floor_by_name[name] = floor
            streak = self._streak_by_name[name]
        else:
            raise ValueError(f"unknown give_up_mode: {self.give_up_mode!r}")
        return streak, streak >= _STUCK_CANDIDATE_GIVE_UP_STREAK


def _fill_exposed_collectives_with_heavy_compute(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
    give_up_mode: str = "streak",
    is_while_body: bool = False,
) -> tuple[bool, list, dict, dict]:
    """Pull heavy compute instructions backward into earlier collectives'
    still-exposed [start, done) windows, to help hide their latency.

    Runs an outer fixed-point loop (up to _MAX_FILL_FIXED_POINT_ITERS
    passes): each pass builds a fresh bookkeeping list of EVERY collective
    in `comp`, recomputing deficit (unhidden latency) from the real,
    current overlap -- not just the ones that looked exposed at the start
    of this function -- ordered earliest-start-first, and only proceeds to
    a next pass if the previous one actually relocated/chased something.
    This matters because both the direct-placement scan and the
    blocker-chase fallbacks below can, as a side effect of fixing one
    window, relocate instructions that happened to be covering a
    *different* window -- including one that was already fully hidden and
    so wasn't even in this pass's worklist. Without rebuilding the full
    worklist from scratch every pass, such newly-exposed collectives would
    never get revisited within this function call at all, and would only
    surface already-exposed on the next (much more expensive,
    schedule.update()-mediated) _run_phase1_to_fixed_point iteration --
    confirmed happening in practice (job 3181428: call-start.2, .4, .38,
    and all-gather-start.8.g2 went from fully hidden to newly exposed
    purely as a side effect of other windows' chases, invisible to a
    single-snapshot worklist).

    For each open window, scans forward from its `done` instruction for a
    heavy compute instruction (a real kernel -- not a trivial
    bitcast/reshape/elementwise op -- with profiled cost at or above
    _HEAVY_COMPUTE_MIN_US) whose own real dependencies (found the same way
    _earliest_legal_pos finds them for a collective start) already sit at
    or before the window's `done` instruction, and relocates it into the
    window. Moving an instruction to an *earlier* position can never
    violate its own downstream consumers: in any valid schedule a consumer
    already sits after its producer, so it still sits after the new, even
    earlier, position too -- only upstream (operand/control-predecessor)
    dependencies need checking, which _earliest_legal_pos already does.
    Repeats per window (re-deriving that window's own deficit from the
    real overlap sum every inner iteration too, for the same reason --
    see the comment at the top of the inner while loop) until its deficit
    is closed or no more legal candidates remain, then moves on to the
    next window.

    Both scans below only ever consider candidates whose cost doesn't
    exceed the slack of whatever collective(s) currently cover their
    position (see _compute_position_margins) -- i.e. heavy compute that
    isn't providing overlap anyone actually still needs. This keeps every
    direct placement strictly non-destructive: moving something can never
    push another collective's overlap below its own latency, but genuinely
    idle time and merely over-covered time are both fair game. (The
    blocker-chase path is unaffected by this restriction -- it already runs
    every candidate move through _find_best_blocker_position's
    whole-computation net-exposure check, a finer-grained version of the
    same guarantee.)
    """
    changed = False
    # Per-collective deficit as of the end of the previous outer pass. A
    # window whose deficit is unchanged since then was already scanned to
    # exhaustion (deterministic search over unchanged state repeats the
    # same outcome), so its inner while-loop is skipped this pass -- keeps
    # the fixed-point loop from redoing real chase work every iteration.
    prev_deficit_by_name: dict[str, float] = {}

    for _fp_iter in range(_MAX_FILL_FIXED_POINT_ITERS):
        pass_changed = False

        windows: list[dict] = []
        cur_deficit_by_name: dict[str, float] = {}
        # One shared O(n) prefix for this whole worklist rebuild, instead of
        # an O(window size) sum per collective -- matters since this now
        # reruns every outer pass for every collective in start_of_done, not
        # just the currently-exposed ones.
        worklist_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
        for ag_done, ag_start in start_of_done.items():
            if ag_start not in positions or ag_done not in positions:
                continue
            profile_key = _resolve_profile_key(ag_start, comp_by_name)
            latency = _profile_costs.get(profile_key)
            if latency is None or latency <= 0:
                continue
            start_pos = positions[ag_start]
            done_pos = positions[ag_done]
            overlap = worklist_prefix[done_pos] - worklist_prefix[start_pos + 1]
            deficit = latency - overlap
            cur_deficit_by_name[ag_start.name] = deficit
            if deficit > 0:
                is_fsdp = profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES)
                prev = prev_deficit_by_name.get(ag_start.name)
                unchanged_since_last_pass = (
                    _fp_iter > 0 and prev is not None and abs(prev - deficit) < 1e-6
                )
                windows.append({
                    "start": ag_start, "done": ag_done, "deficit": deficit, "is_fsdp": is_fsdp,
                    "latency": latency, "skip": unchanged_since_last_pass,
                })

        prev_deficit_by_name = cur_deficit_by_name

        if not windows:
            break

        windows.sort(key=lambda w: positions[w["start"]])
        excluded = set(start_of_done.keys()) | set(start_of_done.values())
        # "Exposed" means genuinely still under-hidden right now -- only these
        # are candidates for _chase_exposed_collective_blocker_earlier, so we
        # don't disturb collectives that are already adequately overlapped.
        exposed_dones = {w["done"] for w in windows}

        # Two collectives' [start, done) windows can genuinely overlap in
        # the schedule (both async collectives in flight concurrently --
        # confirmed for real call-start pairs, e.g. region_19.46's
        # call-start.56/.58). A single instruction relocated into that
        # shared zone counts toward both windows' overlap for free
        # (window["deficit"] is always recomputed from the real schedule,
        # so this gets picked up automatically). What direct placement
        # doesn't do on its own is *prefer* such double-benefit placements
        # among several legal candidates -- computed once per outer pass
        # (a best-effort snapshot, not re-derived every hop).
        sibling_ranges: dict[str, list[tuple[int, int]]] = {}
        for w in windows:
            s, d = positions[w["start"]], positions[w["done"]]
            for other in windows:
                if other is w:
                    continue
                os_, od = positions[other["start"]], positions[other["done"]]
                if os_ < d and s < od:  # ranges [s, d) and [os_, od) overlap
                    sibling_ranges.setdefault(w["start"].name, []).append((os_, od))

        for window in windows:
            if window["skip"]:
                continue
            window_siblings = sibling_ranges.get(window["start"].name, [])
            chase_hops = 0
            # A chase making partial progress triggers an immediate
            # break+rescan-from-done_pos+1, so a second hop can pick up
            # where the first left off (see
            # _chase_heavy_compute_blocker_chain's docstring). But if the
            # SAME candidate is the first blocked thing hit on every rescan
            # (a deep or unresolvable chain), it can loop indefinitely
            # hammering just that one candidate while other reachable
            # candidates later in scan order never get a turn. Confirmed in
            # job 3184492: dot_product_attention_fwd.33.double_buffer_clone
            # alone consumed 448-636 chase attempts without resolving,
            # while te_gemm_v2_ffi.286-310 in the same computation were
            # essentially never tried. Track whether each candidate's floor
            # is actually improving across rescans (keep going) vs.
            # flat/worse (give up, skip it, let others through) -- strategy
            # controlled by give_up_mode (see _StuckCandidateTracker).
            #
            # NOTE: a round-robin variant (one placement per window per
            # round, interleaved across windows) was tried to address
            # run-to-run allocation instability under PGLE cost jitter
            # (jobs 3190682 vs 3190969), then reverted (job 3191562) --
            # interleaving let one window's chase disrupt another's
            # in-progress chase, causing more total churn than draining
            # each window fully before starting the next.
            stuck_tracker = _StuckCandidateTracker(give_up_mode)
            given_up_candidates: set[str] = set()
            while window["deficit"] > 0:
                done_pos = positions[window["done"]]
                start_pos = positions[window["start"]]
                # Re-derive deficit from the real overlap sum every iteration
                # rather than trusting an incremental "deficit -= cost"
                # counter. A chase invoked below can relocate *other*
                # instructions as a side effect, including ones already
                # inside this window, bumping them back out past `done_pos`
                # -- an incremental counter can't notice coverage being
                # undone and would double-credit the displaced-and-replaced
                # instruction. Shared with the position-margins call right
                # below (one O(n) prefix instead of two).
                iter_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
                window["deficit"] = max(
                    0.0, window["latency"] - (iter_prefix[done_pos] - iter_prefix[start_pos + 1])
                )
                if window["deficit"] <= 0:
                    break
                # Recomputed fresh every iteration -- a prior move in this same
                # while-loop shifts positions and can change which stretches of
                # the schedule are covered and by how much slack.
                # Restricted to the scan range this iteration actually
                # queries (done_pos+1..len(seq)) -- the direct-placement
                # gating checks below (`cost > position_margins.get(i, ...)`)
                # never look outside it. A chase, if one gets triggered
                # below, needs the *unrestricted* margins instead (a
                # blocker's own current position can be anywhere, including
                # well before done_pos+1) -- see the two chase call sites,
                # which recompute a full-range copy just before calling.
                position_margins = _compute_position_margins(
                    start_of_done, positions, seq, comp_by_name, prefix=iter_prefix,
                    min_pos=done_pos + 1,
                )
                # Full-range margins for any chase triggered below, computed
                # lazily (only if a chase actually fires this iteration) and
                # cached for the rest of this same while-iteration -- a
                # single scan can refuse-and-retry several blocked
                # candidates before one either succeeds or the whole
                # iteration gives up, and each attempt needs the same
                # full-range snapshot (see the two chase call sites below),
                # not a fresh O(window-overlap) recompute per attempt.
                # Discarded (None again) next while-iteration since a
                # successful chase mutates positions, making it stale.
                chase_margins = None
                candidate = None
                cand_floor = None
                cand_chain: list = []
                # For FSDP-style (all-gather/reduce-scatter) windows, prefer
                # te_gemm/te_grouped_gemm custom-calls (the MoE GEMMs that
                # dominate compute time in DeepSeek-family models -- see
                # _is_te_gemm_custom_call): scan every legally-reachable
                # te_gemm candidate and pick whichever has the EARLIEST
                # legal floor (floor depends on dependency chain, not
                # current scan position, so a later-encountered candidate
                # can still have an earlier floor). Landing closer to
                # start_pos lets the GEMM overlap more of the collective's
                # transfer time, not just close the deficit sum. Falls
                # through to the unrestricted scan only if no te_gemm is
                # legally reachable. Non-FSDP windows skip straight to the
                # unrestricted scan -- no reason to prefer a GEMM there.
                chased = False
                if window["is_fsdp"]:
                    best_floor = None
                    for i in range(done_pos + 1, len(seq)):
                        inst = seq[i]
                        if inst in excluded:
                            continue
                        if inst.name in given_up_candidates:
                            continue
                        if _is_trivially_movable_inst(inst, comp_by_name):
                            continue
                        if not _is_te_gemm_custom_call(inst, comp_by_name):
                            continue
                        cost = _resolve_inst_cost(inst, comp_by_name)
                        if cost < _HEAVY_COMPUTE_MIN_US:
                            continue
                        if cost > position_margins.get(i, float("inf")):
                            # Relocating this would push some other collective's
                            # overlap below its own latency -- skip it so a
                            # direct placement (below) can never silently
                            # expose a window that's currently fine (even one
                            # with lots of slack -- just not THIS much).
                            continue
                        floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
                        if floor > done_pos:
                            streak, give_up_now = stuck_tracker.note_and_maybe_give_up(
                                inst.name, floor
                            )
                            if give_up_now:
                                given_up_candidates.add(inst.name)
                                _logger.debug(
                                    "collective_overlap_pass [%s]: FSDP scan "
                                    "for window %s: giving up on %s after %d "
                                    "consecutive unresolved chase restarts "
                                    "(mode=%s, floor stuck at %d) -- skipping "
                                    "it for the rest of this window's scan.",
                                    module_name, window["start"].name, inst.name,
                                    streak, give_up_mode, floor,
                                )
                            # The candidate's own floor is pinned past our
                            # window -- if that's specifically because it data-
                            # depends on another still-exposed collective's
                            # done sitting in between, try shrinking that gap by
                            # chasing that collective's start earlier instead of
                            # just giving up on this candidate.
                            if blocker is not None and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
                                chased, seq, positions, name_to_pos = (
                                    _chase_exposed_collective_blocker_earlier(
                                        blocker, start_of_done, exposed_dones,
                                        start_pos, done_pos, seq, schedule, comp,
                                        positions, name_to_pos, comp_by_name,
                                    )
                                )
                                if chased:
                                    chase_hops += 1
                                    changed = True
                                    pass_changed = True
                                    _logger.info(
                                        "collective_overlap_pass [%s]: chased "
                                        "exposed-collective blocker %s earlier "
                                        "to unblock heavy-compute candidate %s "
                                        "for window %s (deficit %.1f us "
                                        "remaining).",
                                        module_name, start_of_done[blocker].name,
                                        inst.name, window["start"].name,
                                        window["deficit"],
                                    )
                                    break
                            # Blocker wasn't an exposed collective -- try
                            # chasing it as an arbitrary movable heavy op
                            # instead (e.g. another GEMM the candidate
                            # transitively depends on). Gated on the same
                            # chase_hops budget as the collective chase above.
                            if (
                                not chased
                                and _MAX_HEAVY_COMPUTE_CHASE_HOPS > 0
                                and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS
                            ):
                                # Full-range margins (lazy per-iteration
                                # cache set up above), not the
                                # done_pos+1-restricted `position_margins`
                                # above -- the chase's margin-safe fallback
                                # (_try_relocate_blocker_earlier path 2)
                                # queries at the blocker's *current*
                                # position, which can be anywhere, including
                                # well before done_pos+1.
                                if chase_margins is None:
                                    chase_margins = _compute_position_margins(
                                        start_of_done, positions, seq, comp_by_name,
                                    )
                                heavy_chased, seq, positions, name_to_pos, _, _, _ = (
                                    _chase_heavy_compute_blocker_chain(
                                        inst, comp, schedule, seq, positions, name_to_pos,
                                        comp_by_name, window["start"], done_pos,
                                        window["latency"], start_of_done,
                                        _MAX_HEAVY_COMPUTE_CHASE_HOPS, module_name,
                                        chase_margins, is_while_body=is_while_body,
                                    )
                                )
                                if heavy_chased:
                                    chase_hops += 1
                                    chased = True
                                    changed = True
                                    pass_changed = True
                                    _logger.info(
                                        "collective_overlap_pass [%s]: chased "
                                        "heavy-compute blocker chain to try to "
                                        "unblock candidate %s for window %s "
                                        "(deficit %.1f us remaining).",
                                        module_name, inst.name, window["start"].name,
                                        window["deficit"],
                                    )
                                    break
                            continue
                        if best_floor is None or floor < best_floor:
                            best_floor = floor
                            candidate = inst
                            cand_floor = floor
                            cand_chain = chain

                if chased:
                    # positions shifted under us -- re-derive done_pos/start_pos
                    # and rescan this window fresh next iteration.
                    _logger.debug(
                        "collective_overlap_pass [%s]: DIAG FSDP scan for "
                        "window %s diverted into a chase (chase_hops=%d/%d) -- "
                        "restarting this window's scan from scratch instead of "
                        "reaching the unrestricted scan this pass.",
                        module_name, window["start"].name, chase_hops,
                        _MAX_PRODUCER_RELOCATE_HOPS,
                    )
                    continue

                if candidate is None:
                    _logger.debug(
                        "collective_overlap_pass [%s]: DIAG unrestricted scan "
                        "reached for window %s (done_pos=%d, is_fsdp=%s, "
                        "chase_hops=%d/%d).",
                        module_name, window["start"].name, done_pos,
                        window["is_fsdp"], chase_hops, _MAX_PRODUCER_RELOCATE_HOPS,
                    )
                    unrestricted_chased = False
                    # Collect every legal candidate instead of stopping at
                    # the first -- taking the first-found regardless of size
                    # can badly overshoot a small deficit (e.g. spending a
                    # 960us GEMM to close a 400us gap), stranding the
                    # surplus and making it unavailable to a different
                    # window that needed exactly that much. Best-fit is
                    # picked below, after the loop, from everything legally
                    # reachable this pass. Chase attempts still break out
                    # and restart immediately; only direct placement defers.
                    legal_candidates: list[tuple] = []
                    for i in range(done_pos + 1, len(seq)):
                        inst = seq[i]
                        if inst in excluded:
                            continue
                        if inst.name in given_up_candidates:
                            continue
                        if _is_trivially_movable_inst(inst, comp_by_name):
                            continue
                        cost = _resolve_inst_cost(inst, comp_by_name)
                        if cost < _HEAVY_COMPUTE_MIN_US:
                            continue
                        if cost > position_margins.get(i, float("inf")):
                            continue
                        floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
                        if floor > done_pos:
                            streak, give_up_now = stuck_tracker.note_and_maybe_give_up(
                                inst.name, floor
                            )
                            if give_up_now:
                                given_up_candidates.add(inst.name)
                                _logger.debug(
                                    "collective_overlap_pass [%s]: unrestricted "
                                    "scan for window %s: giving up on %s after "
                                    "%d consecutive unresolved chase restarts "
                                    "(mode=%s, floor stuck at %d) -- skipping "
                                    "it for the rest of this window's scan.",
                                    module_name, window["start"].name, inst.name,
                                    streak, give_up_mode, floor,
                                )
                            # Same two-tier chase as the te_gemm-priority scan
                            # above: exposed-collective case first, then fall
                            # back to an arbitrary movable heavy-compute
                            # blocker chain.
                            if blocker is not None and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
                                unrestricted_chased, seq, positions, name_to_pos = (
                                    _chase_exposed_collective_blocker_earlier(
                                        blocker, start_of_done, exposed_dones,
                                        start_pos, done_pos, seq, schedule, comp,
                                        positions, name_to_pos, comp_by_name,
                                    )
                                )
                                if unrestricted_chased:
                                    chase_hops += 1
                                    changed = True
                                    pass_changed = True
                                    _logger.info(
                                        "collective_overlap_pass [%s]: chased "
                                        "exposed-collective blocker %s earlier "
                                        "to unblock heavy-compute candidate %s "
                                        "for window %s (deficit %.1f us "
                                        "remaining).",
                                        module_name, start_of_done[blocker].name,
                                        inst.name, window["start"].name,
                                        window["deficit"],
                                    )
                                    break
                            if (
                                not unrestricted_chased
                                and _MAX_HEAVY_COMPUTE_CHASE_HOPS > 0
                                and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS
                            ):
                                # Full-range margins (lazy per-iteration
                                # cache) -- see the comment at the FSDP
                                # scan's equivalent chase call above.
                                if chase_margins is None:
                                    chase_margins = _compute_position_margins(
                                        start_of_done, positions, seq, comp_by_name,
                                    )
                                unrestricted_chased, seq, positions, name_to_pos, _, _, _ = (
                                    _chase_heavy_compute_blocker_chain(
                                        inst, comp, schedule, seq, positions, name_to_pos,
                                        comp_by_name, window["start"], done_pos,
                                        window["latency"], start_of_done,
                                        _MAX_HEAVY_COMPUTE_CHASE_HOPS, module_name,
                                        chase_margins, is_while_body=is_while_body,
                                    )
                                )
                                if unrestricted_chased:
                                    chase_hops += 1
                                    changed = True
                                    pass_changed = True
                                    _logger.info(
                                        "collective_overlap_pass [%s]: chased "
                                        "heavy-compute blocker chain to try to "
                                        "unblock candidate %s for window %s "
                                        "(deficit %.1f us remaining).",
                                        module_name, inst.name, window["start"].name,
                                        window["deficit"],
                                    )
                                    break
                            continue
                        legal_candidates.append((inst, floor, chain, cost))

                    if unrestricted_chased:
                        # positions shifted under us -- re-derive done_pos/
                        # start_pos and rescan this window fresh next iteration.
                        continue

                    if legal_candidates:
                        def _lands_in_sibling(floor: int) -> bool:
                            return any(s <= floor < d for s, d in window_siblings)

                        remaining = window["deficit"]
                        sufficient = [c for c in legal_candidates if c[3] >= remaining]
                        if sufficient:
                            # Smallest candidate that still fully closes the
                            # deficit (minimizes overshoot/waste), preferring
                            # one that also lands inside a sibling window's
                            # range so one relocation counts toward both.
                            candidate, cand_floor, cand_chain, _ = min(
                                sufficient,
                                key=lambda c: (not _lands_in_sibling(c[1]), c[3]),
                            )
                        else:
                            # Nothing alone reaches the deficit -- take the
                            # largest available, same sibling preference.
                            candidate, cand_floor, cand_chain, _ = max(
                                legal_candidates,
                                key=lambda c: (_lands_in_sibling(c[1]), c[3]),
                            )

                if candidate is None:
                    break

                target = max(cand_floor, start_pos + 1)
                to_move_set = set(cand_chain) | {candidate}
                new_seq = [inst for inst in seq if inst not in to_move_set]
                ins_pos = target
                for inst in cand_chain:  # already in topological (schedule) order
                    new_seq.insert(ins_pos, inst)
                    ins_pos += 1
                new_seq.insert(ins_pos, candidate)
                schedule.set_sequence(comp, new_seq)
                seq = new_seq
                positions = {inst: i for i, inst in enumerate(seq)}
                name_to_pos = {inst.name: i for inst, i in positions.items()}
                changed = True
                pass_changed = True

                cost = _resolve_inst_cost(candidate, comp_by_name)
                prev_deficit = window["deficit"]
                # Not claiming a post-relocation deficit here: a naive
                # "deficit -= cost" can double-credit a candidate displaced
                # and re-placed within the same loop (see top-of-loop
                # recompute above). Next iteration's recompute is the
                # authoritative post-relocation figure.
                _logger.info(
                    "collective_overlap_pass [%s]: relocated heavy compute %s "
                    "(cost %.1f us) into exposed window of %s (pre-relocation "
                    "deficit was %.1f us; see next deficit= line for the real "
                    "post-relocation figure).",
                    module_name, candidate.name, cost, window["start"].name,
                    prev_deficit,
                )
        if not pass_changed:
            break
    return changed, seq, positions, name_to_pos


# ---------------------------------------------------------------------------
# Phase 1: schedule reordering
# ---------------------------------------------------------------------------
_WHILE_CALLS_RE = re.compile(r"(?:condition|body)=%([A-Za-z0-9_.]+)")


def _innermost_first_computations(module, schedule) -> tuple[list, set]:
    """Return (computations, while_body_comps).

    computations are non-fusion scheduled computations in innermost-first
    DFS order: while-body computations are visited before the computation
    that contains their while instruction, so that inner schedule changes
    are committed before outer schedules are processed.  Nested while loops
    are handled by recursing depth-first.  The entry computation is always
    last.

    while_body_comps is the set of computations called as a while body
    (i.e. every scheduled computation except the true module entry) --
    exposed so callers can tell "am I working on the entry computation or
    a while loop" without recomputing the same while-call scan themselves.
    """
    all_comps = [
        c for c in module.make_nonfusion_computations()
        if schedule.sequence(c) is not None
    ]
    # Local name->comp map restricted to this same accessor call, so lookups
    # stay identity-consistent with all_comps (HloComputation has no custom
    # __eq__/__hash__, so objects from a different accessor call may not
    # compare equal even for the same underlying computation).
    local_comp_by_name = {c.name: c for c in all_comps}

    # Build computation -> while-body children, and the set of all
    # while-body callees (the entry computation is the one with no callers).
    # inst.opcode is a jaxlib._hlo.HloOpcode enum and HloInstruction has no
    # called_computations() accessor, so both checks go through
    # _opcode_str()/to_string() parsing instead of direct attribute access.
    children: dict = {c: [] for c in all_comps}
    all_callees: set = set()
    for comp in all_comps:
        for inst in schedule.sequence(comp):
            if _opcode_str(inst) != "while":
                continue
            try:
                text = inst.to_string()
            except Exception:
                continue
            for name in _WHILE_CALLS_RE.findall(text):
                called = local_comp_by_name.get(name)
                if called is not None and schedule.sequence(called) is not None:
                    children[comp].append(called)
                    all_callees.add(called)

    # The entry computation is the only one not called as a while body.
    roots = [c for c in all_comps if c not in all_callees]
    if not roots:
        # Fallback: return in original order (no while loop structure found).
        return all_comps, all_callees

    def _collect(comp, visited: set, result: list) -> None:
        if comp in visited:
            return
        visited.add(comp)
        for child in children.get(comp, []):
            _collect(child, visited, result)
        result.append(comp)

    visited: set = set()
    result: list = []
    for root in roots:
        _collect(root, visited, result)
    # Any computations not reachable from a root (e.g. orphaned while bodies).
    for comp in all_comps:
        if comp not in visited:
            result.append(comp)
    return result, all_callees


_WHILE_BODY_RE = re.compile(r"\bbody=%([A-Za-z0-9_.]+)")


def _log_profile_coverage_gaps(module, schedule, module_name: str) -> None:
    """Diagnostic: log fusion/custom-call instructions inside while-loop
    bodies that have NO entry in _profile_costs.

    _profile_costs is the PGLE-derived cost table every heavy-compute
    decision in this pass relies on (deficit math,
    _fill_exposed_collectives_with_heavy_compute, blocker relocation). A
    missing entry is silently treated as cost 0.0 -- i.e. free -- by every
    one of those `.get(name, 0.0)` lookups, with no error or warning
    otherwise, which can hide a genuinely expensive op from every
    heavy-compute code path.
    """
    all_comps = [
        c for c in module.make_nonfusion_computations() if schedule.sequence(c) is not None
    ]
    comp_by_name = {c.name: c for c in all_comps}
    while_bodies: set = set()
    for comp in all_comps:
        for inst in schedule.sequence(comp):
            if _opcode_str(inst) != "while":
                continue
            try:
                text = inst.to_string()
            except Exception:
                continue
            for name in _WHILE_BODY_RE.findall(text):
                body = comp_by_name.get(name)
                if body is not None:
                    while_bodies.add(body)

    if not while_bodies:
        return

    missing: list[str] = []
    present = 0
    total = 0
    for comp in while_bodies:
        for inst in schedule.sequence(comp):
            opc = _opcode_str(inst)
            if opc not in ("fusion", "custom-call"):
                continue
            total += 1
            if inst.name in _profile_costs:
                present += 1
            else:
                missing.append(f"{inst.name}({opc})")

    _logger.info(
        "collective_overlap_pass [%s]: profile coverage in %d while-body "
        "computation(s): %d/%d fusion/custom-call instructions have a "
        "PGLE cost entry (%d missing).",
        module_name, len(while_bodies), present, total, len(missing),
    )
    if missing:
        _logger.info(
            "collective_overlap_pass [%s]: missing-profile instructions "
            "(first 150 of %d): %s",
            module_name, len(missing), ", ".join(missing[:150]),
        )


def _fill_exposed_collectives_best_of(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
    is_while_body: bool = False,
) -> tuple[bool, list, dict, dict]:
    """Try every _GIVE_UP_MODES strategy for
    _fill_exposed_collectives_with_heavy_compute against this computation,
    each starting from the same baseline schedule, and keep whichever
    produces the lowest real total exposed time (_total_exposed_us) across
    this computation's own collectives.

    No single give-up strategy dominated across every computation observed
    in practice: for the small while-loop-body computations (a handful of
    call-start windows sharing a scarce, mostly-unreachable pool of heavy
    compute -- see the dot_product_attention_fwd.33.double_buffer_clone
    investigation), "streak" won by a wide margin (job 3184713). For the
    much larger ENTRY computation, "floor_aware" recovered significantly
    more of the fixed-point loop's improvement (job 3187886) because it
    doesn't prematurely abandon candidates whose multi-hop chains take more
    than a few hops to genuinely converge. Rather than pick one heuristic
    and accept whichever trade-off it implies everywhere, just measure and
    keep the best per computation -- this is cheap to do since each
    computation is scheduled independently anyway.

    Costs roughly len(_GIVE_UP_MODES)x the compute of a single strategy for
    this comp's fill step specifically (not the whole pass) -- acceptable
    as a one-time cost, but _run_phase1_to_fixed_point re-enters
    _phase1_reorder (and so this function, once per computation) up to
    _MAX_PHASE1_FIXED_POINT_ITERS times per _compute_collective_overlap
    call, and re-running the full comparison every single time compounded
    into a job timeout in practice (3188145). So the winning mode is cached
    per computation name (_fill_strategy_cache) after the first comparison
    within a _run_phase1_to_fixed_point call, and subsequent re-entries for
    the same computation just replay that cached strategy with a single
    trial instead of re-comparing all of them from scratch.
    """
    global _fill_strategy_cache
    cached_mode = _fill_strategy_cache.get(comp.name)
    if cached_mode is not None:
        # Log _total_exposed_us on the cached path too (not just the first,
        # full-comparison call) -- otherwise every re-entry after the first
        # for this comp goes dark, and drift from later schedule.update()/
        # verify() calls (well-documented elsewhere in this file -- see
        # _run_phase1_to_fixed_point's docstring) between _phase1_reorder
        # re-entries becomes invisible instead of a visible trajectory.
        pre_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
        changed, seq, positions, name_to_pos = _fill_exposed_collectives_with_heavy_compute(
            list(seq), schedule, comp, start_of_done, dict(positions), dict(name_to_pos),
            comp_by_name, module_name, give_up_mode=cached_mode, is_while_body=is_while_body,
        )
        post_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
        _logger.info(
            "collective_overlap_pass [%s]: comp %s: cached fill strategy "
            "'%s' re-applied -- total exposed on entry %.1f us -> %.1f us "
            "after this re-entry's fill (changed=%s).",
            module_name, comp.name, cached_mode, pre_total, post_total, changed,
        )
        return changed, seq, positions, name_to_pos

    baseline_seq = list(seq)
    baseline_positions = dict(positions)
    baseline_name_to_pos = dict(name_to_pos)

    best = None  # (total_exposed, changed, seq, positions, name_to_pos, mode)
    for mode in _GIVE_UP_MODES:
        schedule.set_sequence(comp, list(baseline_seq))
        trial_changed, trial_seq, trial_positions, trial_name_to_pos = (
            _fill_exposed_collectives_with_heavy_compute(
                list(baseline_seq), schedule, comp, start_of_done,
                dict(baseline_positions), dict(baseline_name_to_pos),
                comp_by_name, module_name, give_up_mode=mode, is_while_body=is_while_body,
            )
        )
        total = _total_exposed_us(start_of_done, trial_positions, trial_seq, comp_by_name)
        _logger.info(
            "collective_overlap_pass [%s]: fill strategy '%s' for comp %s: "
            "total exposed %.1f us (changed=%s).",
            module_name, mode, comp.name, total, trial_changed,
        )
        if best is None or total < best[0] - 1e-6:
            best = (total, trial_changed, trial_seq, trial_positions, trial_name_to_pos, mode)

    total, changed, seq, positions, name_to_pos, mode = best
    schedule.set_sequence(comp, seq)
    _fill_strategy_cache[comp.name] = mode
    _logger.info(
        "collective_overlap_pass [%s]: comp %s: selected fill strategy "
        "'%s' (total exposed %.1f us).",
        module_name, comp.name, mode, total,
    )
    # A per-window retry of stuck windows (e.g. all-gather-start.8.g0/.g3
    # under 'none', job 3191924) with the other give_up modes was tried and
    # reverted (job 3193061): every alternate mode converged to the exact
    # same deficit. Root cause: te_gemm_v2_ffi.81/.87 (dot_product_attention
    # _fwd.10's direct producers) both take all-gather-done.8.g2 as an
    # operand, a real data-dependency floor no give_up policy can move. This
    # is a consequence of how phase 2's split grouped all-gather-start.8's
    # operands into g0-g3, not a fill-algorithm gap -- would need a
    # different split partitioning to address.
    return changed, seq, positions, name_to_pos


# DIAGNOSTIC (temporary): module-wide accumulator of "already hidden"
# verdicts, so _compute_collective_overlap can re-verify them again after
# schedule.update()/module.set_schedule(), not just within the comp that
# produced each verdict. Cleared at the top of each _phase1_reorder call.
_diag_hidden_windows: list[dict] = []

# Cache of which _GIVE_UP_MODES strategy won _fill_exposed_collectives_best_of's
# comparison for a given computation, so the ~3x-cost multi-strategy trial
# only runs once per computation per _run_phase1_to_fixed_point call instead
# of on every re-entry into _phase1_reorder (re-comparing on every re-entry
# caused a timeout -- job 3188145). Cleared at the top of
# _run_phase1_to_fixed_point since profiled costs/positions can differ
# across separate _compute_collective_overlap invocations.
_fill_strategy_cache: dict[str, str] = {}


def _may_hoist_control_pred(pred, new_pos, seq, positions, comp_by_name) -> bool:
    """Whether control-predecessor `pred` may move earlier to new_pos.

    No cost or opcode criterion: an FSDP start's control-predecessors are
    hoisted to their own earliest legal position whatever they are. The one
    guard is for an async-done, since moving it earlier shrinks its own
    start's window: refused if that start has a profiled latency the shrunken
    window no longer covers.
    """
    opc = _opcode_str(pred)
    if not _HOIST_ASYNC_DONE_GUARD or not (opc == "async-done" or opc.endswith("-done")):
        return True
    ops = list(pred.operands())
    start = ops[0] if ops else None
    if start is None or start not in positions:
        return False
    try:
        latency = _profile_costs.get(_resolve_profile_key(start, comp_by_name), 0.0)
    except Exception:
        latency = 0.0
    if not latency or latency <= 0:
        return True
    remaining = sum(
        _resolve_inst_cost(seq[i], comp_by_name)
        for i in range(positions[start] + 1, new_pos)
    )
    return remaining >= latency


def _is_te_ep_call(inst, comp_by_name: dict) -> bool:
    """True if inst is any te_ep async call-start (prepare, dispatch or combine)."""
    if not _is_async_start(inst):
        return False
    return _resolve_profile_key(inst, comp_by_name).startswith(_TE_EP_PREFIX)


def _fsdp_te_ep_defer_target(
    ag_start, ag_done, floor, seq, positions, comp_by_name, latency, committed,
):
    """Index in seq to place an FSDP start before, or None for "earliest legal position".

    All NCCL work (FSDP collectives and te_ep calls) shares one stream, so an
    FSDP start launched ahead of a te_ep call delays that call, and compute that
    needs its result sits idle (job 3212382: all-gather.142 and
    reduce-scatter.50/.51 ran back to back ahead of te_ep_prepare_ffi.12, leaving
    the compute stream idle ~3.5 ms). So:
      1. If the heavy compute between the floor and the first te_ep call can still
         cover this collective (after the latency of FSDP collectives already
         placed there), start it at the floor.
      2. Otherwise start it right before the first heavy compute after that te_ep
         call that depends on a te_ep call-done and is followed by enough heavy
         compute to cover its latency.
      3. Otherwise None.
    """
    if not latency or latency <= 0:
        return None
    done_pos = positions[ag_done]
    first_te = next(
        (i for i in range(floor, done_pos) if _is_te_ep_call(seq[i], comp_by_name)), None,
    )
    if first_te is None:
        return None

    def heavy(i) -> float:
        inst = seq[i]
        if _is_async_start(inst) or _opcode_str(inst).endswith("-done"):
            return 0.0
        cost = _resolve_inst_cost(inst, comp_by_name)
        return cost if cost >= _HEAVY_COMPUTE_MIN_US else 0.0

    key = seq[first_te].name
    available = sum(heavy(i) for i in range(floor, first_te)) - committed.get(key, 0.0)
    if available >= latency:
        committed[key] = committed.get(key, 0.0) + latency
        return None

    te_done = set()
    for inst in seq:
        if _is_te_ep_call(inst, comp_by_name):
            te_done.update(inst.users())
    depends: dict = {}
    for inst in seq:
        depends[inst] = inst in te_done or any(depends.get(op, False) for op in inst.operands())
    suffix = [0.0] * (done_pos + 1)
    for i in range(done_pos - 1, first_te, -1):
        suffix[i] = suffix[i + 1] + heavy(i)
    for i in range(first_te + 1, done_pos):
        if heavy(i) > 0 and depends[seq[i]] and suffix[i] >= latency:
            return i
    return None


def _simulate_comm_stream(order, cost: dict, lat: dict, done_start: dict):
    """Simulate compute (in `order`) against one FIFO communication stream.

    Compute ops run back to back at their profiled cost. Every async start is
    queued on the comm stream and begins at the later of its issue time and the
    end of the previous queued op; an async done stalls compute until its start
    has finished. Returns (total stall us, {done: stall us}).

    NCCL collectives and te_ep calls share one stream, so a collective issued
    behind a te_ep call can't start until that call's kernels finish, and a
    te_ep call issued behind a collective waits for it (job 3212637). The
    static window cost used elsewhere doesn't see either effect.
    """
    t = 0.0
    comm = 0.0
    end: dict = {}
    total = 0.0
    stalls: dict = {}
    for inst in order:
        start = done_start.get(inst)
        if start is not None:
            e = end.get(start)
            if e is not None and e > t:
                stalls[inst] = e - t
                total += e - t
                t = e
        elif inst in lat:
            begin = t if t > comm else comm
            comm = begin + lat[inst]
            end[inst] = comm
        else:
            t += cost.get(inst, 0.0)
    return total, stalls


def _best_fsdp_placement(
    ag_start, ag_done, floor, to_move, seq, positions, cost, lat, done_start, ctrl_succs,
):
    """Choose where to put an FSDP start (and its done) to minimize simulated stalls.

    Candidates are insertion points for the start between its earliest legal
    position and the first consumer of its done. The done may stay put or move
    to just before its first consumer, so a start placed after its current
    done shifts the done along with it. Both stay within topological order:
    the start after its operands and control-predecessors, the done after the
    start and before its consumers and control-successors. A start moving
    earlier brings its trivial operand chain (`to_move`); one moving later
    leaves the chain where it is.

    Ties prefer leaving the done alone, then the earliest start. Returns
    (key, early, start_idx, done_idx, base) or None, where indices are into
    `base` (seq without the moved instructions) and key[0] is the simulated
    total stall.
    """
    start_pos = positions[ag_start]
    done_pos = positions[ag_done]
    consumers = [positions[u] for u in ag_done.users() if u in positions]
    consumers += [positions[x] for x in ctrl_succs.get(ag_done.name, []) if x in positions]
    if not consumers:
        return None
    user_pos = min(consumers)
    p_max = min(
        [user_pos]
        + [positions[x] for x in ctrl_succs.get(ag_start.name, []) if x in positions and x is not ag_done]
    )
    best = None
    for early in (True, False):
        rem = {ag_start, ag_done} | (set(to_move) if early else set())
        removed_before = [0] * (len(seq) + 1)
        count = 0
        for i, inst in enumerate(seq):
            removed_before[i] = count
            if inst in rem:
                count += 1
        removed_before[len(seq)] = count
        base = [inst for inst in seq if inst not in rem]
        lo, hi = (floor, start_pos) if early else (start_pos + 1, p_max)
        hi = min(hi, p_max)
        q_latest = user_pos - removed_before[user_pos]
        q_cur = done_pos - removed_before[done_pos]
        for i in range(lo, hi + 1):
            if i < len(seq) and i not in (lo, hi, start_pos):
                inst = seq[i]
                if not (
                    cost.get(inst, 0.0) >= _HEAVY_COMPUTE_MIN_US
                    or inst in lat
                    or inst in done_start
                ):
                    continue
            pb = i - removed_before[i] if i < len(seq) else len(base)
            for q in {q_latest, q_cur}:
                if q < pb:
                    continue
                order = base[:pb] + [ag_start] + base[pb:q] + [ag_done] + base[q:]
                total, _ = _simulate_comm_stream(order, cost, lat, done_start)
                key = (round(total, 1), q != q_cur, pb)
                if best is None or key < best[0]:
                    best = (key, early, pb, q, base)
    return best


# Recursion bound when hoisting a control-predecessor whose own latest operand
# must move first.
_MAX_PRED_HOIST_DEPTH = 8


def _hoist_fsdp_starts_to_floor(
    seq, schedule, comp, start_of_done, positions, name_to_pos, comp_by_name, module_name,
):
    """Move each FSDP start in a while body to its earliest legal position.

    Runs after the fill step. The per-collective loop in _phase1_reorder skips
    a collective once its static window looks hidden, and later fill/chase
    moves can then land compute ahead of it that it has no dependency on
    (job 3204509's all-gather-start.9 sat after te_grouped_quantize_ffi.177.
    double_buffer_clone and the dynamic_slice_fusion.25 GEMM). Its operand
    slices carry control-predecessors (loop_add_fusion.9/10, bitcasts of
    earlier all-gather-dones, DUS/async-done fusions) whose current positions
    set the floor even though they could themselves sit much earlier. So while
    a control-predecessor pins the floor, move that predecessor to its
    own earliest legal position (after its operands and control-predecessors),
    then move the start and its trivial operand chain. The done stays put, so
    the window only grows.
    """
    changed = False

    def _move_earlier(inst, depth) -> bool:
        """Move inst to just after its latest operand/control-predecessor.

        If it has no room because that latest one sits right before it (e.g.
        a bitcast chain), hoist that one first, recursively up to `depth`.
        """
        nonlocal seq, positions, name_to_pos, changed
        pos = positions[inst]
        floor, binder = 0, None
        for op in inst.operands():
            p = positions.get(op)
            if p is not None and p + 1 > floor:
                floor, binder = p + 1, op
        for name in _control_predecessor_names(inst):
            p = name_to_pos.get(name)
            if p is not None and p + 1 > floor:
                floor, binder = p + 1, seq[p]
        if floor < pos:
            if not _may_hoist_control_pred(inst, floor, seq, positions, comp_by_name):
                _logger.debug(
                    "collective_overlap_pass [%s]: HOISTDIAG %s (pos %d, floor %d): refused by async-done guard.",
                    module_name, inst.name, pos, floor,
                )
                return False
            moving = {inst}
            new_seq = [i for i in seq if i not in moving]
            if len(new_seq) != len(seq) - 1:
                _logger.warning(
                    "collective_overlap_pass [%s]: HOISTDIAG %s not found exactly once in sequence; skipping move.",
                    module_name, inst.name,
                )
                return False
            new_seq.insert(floor, inst)
            schedule.set_sequence(comp, new_seq)
            seq = new_seq
            positions = {i: k for k, i in enumerate(seq)}
            name_to_pos = {i.name: k for i, k in positions.items()}
            changed = True
            return True
        if binder is None or depth <= 0:
            _logger.debug(
                "collective_overlap_pass [%s]: HOISTDIAG %s (pos %d, floor %d): %s.",
                module_name, inst.name, pos, floor,
                "no operand/control-predecessor to hoist" if binder is None else "depth limit reached",
            )
            return False
        _logger.debug(
            "collective_overlap_pass [%s]: HOISTDIAG %s (pos %d): no room, latest dependency is %s (pos %d, %s); hoisting it first.",
            module_name, inst.name, pos, binder.name, positions[binder], _opcode_str(binder),
        )
        if not _move_earlier(binder, depth - 1):
            return False
        return _move_earlier(inst, depth - 1)

    te_ep_committed: dict = {}
    sim_cost: dict = {}
    sim_lat: dict = {}
    ctrl_succs: dict = {}
    if _FSDP_STREAM_MODEL:
        for inst in seq:
            if _is_async_start(inst) or _opcode_str(inst).endswith("-done"):
                sim_cost[inst] = 0.0
            else:
                sim_cost[inst] = _resolve_inst_cost(inst, comp_by_name)
            for name in _control_predecessor_names(inst):
                ctrl_succs.setdefault(name, []).append(inst)
        for d_inst, s_inst in start_of_done.items():
            try:
                sim_lat[s_inst] = _profile_costs.get(_resolve_profile_key(s_inst, comp_by_name), 0.0) or 0.0
            except Exception:
                sim_lat[s_inst] = 0.0
        _logger.info(
            "collective_overlap_pass [%s]: stream model: %s simulated total exposed %.1f us before FSDP placement.",
            module_name, comp.name, _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done)[0],
        )
    for ag_done, ag_start in sorted(start_of_done.items(), key=lambda kv: positions[kv[1]]):
        profile_key = _resolve_profile_key(ag_start, comp_by_name)
        if not profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES):
            continue
        orig_pos = positions[ag_start]
        for _ in range(_MAX_PRODUCER_RELOCATE_HOPS):
            cp_pins: list = []
            floor, to_move, _blk = _earliest_legal_pos(
                ag_start, positions, name_to_pos, comp_by_name, cp_pins,
            )
            if not cp_pins:
                break
            pin_pos, pin_name = max(cp_pins)
            if pin_pos + 1 < floor or pin_pos + 1 < 1:
                break  # data dependency pins the floor, not a control-predecessor
            pred = seq[pin_pos]
            if not _move_earlier(pred, _MAX_PRED_HOIST_DEPTH):
                _logger.debug(
                    "collective_overlap_pass [%s]: %s floor %d pinned by control-predecessor %s "
                    "(pos %d) which could not be hoisted.",
                    module_name, ag_start.name, floor, pin_name, pin_pos,
                )
                break
            _logger.debug(
                "collective_overlap_pass [%s]: hoisted control-predecessor %s of %s from pos %d.",
                module_name, pin_name, ag_start.name, pin_pos,
            )
        floor, to_move, _blk = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
        start_pos = positions[ag_start]
        if _FSDP_STREAM_MODEL and sim_lat.get(ag_start, 0.0) > 0:
            before = _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done)[0]
            best = _best_fsdp_placement(
                ag_start, ag_done, floor, to_move, seq, positions, sim_cost, sim_lat,
                start_of_done, ctrl_succs,
            )
            _logger.debug(
                "collective_overlap_pass [%s]: STREAMDIAG %s: start pos %d, done pos %d, floor %d, to_move %d, "
                "latency %.1f us, simulated total %.1f us; best %s.",
                module_name, ag_start.name, start_pos, positions[ag_done], floor, len(to_move),
                sim_lat.get(ag_start, 0.0), before,
                "none" if best is None else "total %.1f us at base idx %d, done idx %d, %s" % (
                    best[0][0], best[2], best[3], "chain moved" if best[1] else "chain stays"),
            )
            if best is not None:
                key, early, pb, q, base = best
                if key[0] <= before + 0.05:
                    done_pos_old = positions[ag_done]
                    new_seq = (
                        base[:pb] + (list(to_move) if early else []) + [ag_start]
                        + base[pb:q] + [ag_done] + base[q:]
                    )
                    if len(new_seq) == len(seq) and set(new_seq) == set(seq) and new_seq != seq:
                        schedule.set_sequence(comp, new_seq)
                        seq = new_seq
                        positions = {inst: i for i, inst in enumerate(seq)}
                        name_to_pos = {inst.name: i for inst, i in positions.items()}
                        changed = True
                        _logger.info(
                            "collective_overlap_pass [%s]: stream model placed %s: start pos %d -> %d, done pos %d -> %d "
                            "(simulated total exposed %.1f -> %.1f us).",
                            module_name, ag_start.name, orig_pos, positions[ag_start],
                            done_pos_old, positions[ag_done], before, key[0],
                        )
                    continue
        target = None
        if _FSDP_TE_EP_AWARE:
            target = _fsdp_te_ep_defer_target(
                ag_start, ag_done, floor, seq, positions, comp_by_name,
                _profile_costs.get(profile_key), te_ep_committed,
            )
        if target is not None:
            if target > start_pos:
                # Moving later: stop before any instruction that is control-ordered after the start.
                for k in range(start_pos + 1, target):
                    if ag_start.name in _control_predecessor_names(seq[k]):
                        target = k
                        break
                to_move = []
            if target == start_pos or target == start_pos + 1:
                continue
            to_move_set = set(to_move) | {ag_start}
            new_seq = [inst for inst in seq if inst not in to_move_set]
            ins_pos = sum(1 for inst in seq[:target] if inst not in to_move_set)
            for inst in to_move:
                new_seq.insert(ins_pos, inst)
                ins_pos += 1
            new_seq.insert(ins_pos, ag_start)
            schedule.set_sequence(comp, new_seq)
            seq = new_seq
            positions = {inst: i for i, inst in enumerate(seq)}
            name_to_pos = {inst.name: i for inst, i in positions.items()}
            changed = True
            _logger.info(
                "collective_overlap_pass [%s]: placed %s at pos %d (was %d) before te_ep-dependent heavy compute (te_ep-aware).",
                module_name, ag_start.name, positions[ag_start], orig_pos,
            )
            continue
        if floor + len(to_move) >= start_pos:
            continue
        to_move_set = set(to_move) | {ag_start}
        new_seq = [inst for inst in seq if inst not in to_move_set]
        ins_pos = floor
        for inst in to_move:
            new_seq.insert(ins_pos, inst)
            ins_pos += 1
        new_seq.insert(ins_pos, ag_start)
        schedule.set_sequence(comp, new_seq)
        seq = new_seq
        positions = {inst: i for i, inst in enumerate(seq)}
        name_to_pos = {inst.name: i for inst, i in positions.items()}
        changed = True
        _logger.info(
            "collective_overlap_pass [%s]: hoisted %s from pos %d to %d with %d "
            "relocated operand(s) (final floor pass).",
            module_name, ag_start.name, orig_pos, positions[ag_start], len(to_move),
        )
    return changed, seq, positions, name_to_pos


def _phase1_reorder(module, schedule, module_name: str) -> tuple[bool, list[_SplitCandidate]]:
    """Move async collective starts earlier where latency is under-hidden.

    Returns (changed, split_candidates) where split_candidates contains
    collectives that still had a deficit and were fully dep-blocked.
    """
    global _diag_hidden_windows
    _diag_hidden_windows = []
    changed = False
    split_candidates: list[_SplitCandidate] = []
    # Module-wide name->computation map (includes fusion sub-computations,
    # unlike make_nonfusion_computations()), used to inspect a fusion's body
    # for triviality — see _is_trivial_fusion_body.
    comp_by_name = {c.name: c for c in module.computations()}

    ordered_comps, while_body_comps = _innermost_first_computations(module, schedule)
    for comp in ordered_comps:
        is_while_body = comp in while_body_comps
        if _WHILE_BODY_ONLY and not is_while_body:
            _logger.debug(
                "collective_overlap_pass [%s]: COLLECTIVE_OVERLAP_WHILE_BODY_ONLY=1 "
                "-- skipping entry computation %s entirely.",
                module_name, comp.name,
            )
            continue
        seq = list(schedule.sequence(comp))

        start_of_done: dict = {}
        for inst in seq:
            if _is_async_start(inst):
                users = list(inst.users())
                if len(users) == 1:
                    start_of_done[users[0]] = inst
                else:
                    _logger.debug(
                        "collective_overlap_pass [%s]: async-start %s has "
                        "%d users (expected exactly 1, its done); skipping.",
                        module_name, inst.name, len(users),
                    )

        if not start_of_done:
            continue

        _logger.debug(
            "collective_overlap_pass [%s]: found %d async collective pairs.",
            module_name, len(start_of_done),
        )

        positions = {inst: i for i, inst in enumerate(seq)}
        name_to_pos = {inst.name: i for inst, i in positions.items()}

        # DIAGNOSTIC (temporary): windows marked "already hidden" below are
        # never revisited by this function. Record what justified that
        # verdict here so we can re-check it against the *final* seq/
        # positions for this comp, after the heavy-compute fill step has
        # run -- see the re-verification block right after the
        # _fill_exposed_collectives_with_heavy_compute call below.
        already_hidden_snapshot: list[dict] = []

        for ag_done, ag_start in start_of_done.items():
            profile_key = _resolve_profile_key(ag_start, comp_by_name)
            collective_latency = _profile_costs.get(profile_key)
            if collective_latency is None or collective_latency <= 0:
                _logger.debug(
                    "collective_overlap_pass [%s]: no profile entry for %s "
                    "(resolved profile key: %s).",
                    module_name, ag_start.name, profile_key,
                )
                continue

            ag_start_pos = positions[ag_start]
            ag_done_pos = positions[ag_done]

            window_costs = [
                (seq[i].name, _resolve_inst_cost(seq[i], comp_by_name))
                for i in range(ag_start_pos + 1, ag_done_pos)
            ]
            current_overlap = sum(c for _, c in window_costs)
            if _logger.isEnabledFor(logging.DEBUG):
                top = sorted((c for c in window_costs if c[1] > 0), key=lambda c: -c[1])[:10]
                _logger.debug(
                    "collective_overlap_pass [%s]: %s window [%d, %d) cost "
                    "breakdown (top %d of %d nonzero, total=%.1f us): %s",
                    module_name, ag_start.name, ag_start_pos + 1, ag_done_pos,
                    len(top), sum(1 for _, c in window_costs if c > 0), current_overlap,
                    ", ".join(f"{n}={c:.1f}us" for n, c in top),
                )

            if current_overlap >= collective_latency:
                _logger.debug(
                    "collective_overlap_pass [%s]: %s already hidden "
                    "(overlap=%.1f us >= latency=%.1f us).",
                    module_name, ag_start.name, current_overlap, collective_latency,
                )
                if profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES):
                    done_moved, seq, positions, name_to_pos = _relocate_fsdp_done_before_te_ep(
                        ag_start, ag_done, comp, schedule, seq, positions, name_to_pos,
                        comp_by_name, collective_latency, module_name,
                    )
                    if done_moved:
                        changed = True
                        ag_start_pos = positions[ag_start]
                        ag_done_pos = positions[ag_done]
                _hidden_entry = {
                    "comp": comp,
                    "ag_start": ag_start,
                    "ag_done": ag_done,
                    "start_pos": ag_start_pos,
                    "done_pos": ag_done_pos,
                    "window_names": [n for n, _ in window_costs],
                    "overlap": current_overlap,
                    "latency": collective_latency,
                }
                already_hidden_snapshot.append(_hidden_entry)
                _diag_hidden_windows.append(_hidden_entry)
                continue

            deficit = collective_latency - current_overlap
            _logger.debug(
                "collective_overlap_pass [%s]: %s deficit=%.1f us "
                "(latency=%.1f us, overlap=%.1f us).",
                module_name, ag_start.name, deficit, collective_latency, current_overlap,
            )

            # Find the earliest legally reachable position for ag_start (plus
            # any trivial/zero-cost operand chain that must move with it).
            # We always move all the way to this floor rather than just far
            # enough to close the deficit: any earlier position only adds
            # overlap headroom, and going as early as legally allowed also
            # puts the collective in front of any heavy compute (e.g. GEMMs)
            # that isn't an actual data/control dependency of its own.
            floor, to_move, blocker = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
            orig_ag_start_pos = ag_start_pos  # for the "moved from X to Y" log below

            # For FSDP collectives (all-gather/reduce-scatter), the above
            # floor only ever walks ag_start's own *trivial* operand chain
            # -- it stops cold at the first non-trivial (real) producer,
            # never asking whether that producer itself has room to move.
            # If an unrelated, genuinely heavy instruction (a GEMM) with no
            # data dependency on ag_start at all sits even further back,
            # structurally reachable, ag_start still reports "cannot move"
            # because nothing ever tried relocating its own blocker out of
            # the way first. See _chase_ag_start_blocker_toward_heavy_compute's
            # docstring for the concrete case (job 3204236) that motivated
            # this.
            if profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES):
                (
                    chased_blocker, seq, positions, name_to_pos, floor, to_move, blocker,
                ) = _chase_ag_start_blocker_toward_heavy_compute(
                    ag_start, comp, schedule, seq, positions, name_to_pos, comp_by_name,
                    _MAX_PRODUCER_RELOCATE_HOPS, module_name,
                )
                if chased_blocker:
                    changed = True
                    ag_start_pos = positions[ag_start]
                    ag_done_pos = positions[ag_done]

            # ag_start's own trivial chain lands immediately before it, so
            # its actual post-move position is floor + len(to_move), not
            # floor itself -- comparing against the bare floor would treat
            # an already-optimally-packed ag_start (no gap before it) as
            # movable and perform a no-op "move".
            if floor + len(to_move) >= ag_start_pos:
                _logger.debug(
                    "collective_overlap_pass [%s]: %s cannot move "
                    "(no legal earlier position, deficit=%.1f us).",
                    module_name, ag_start.name, deficit,
                )
                if deficit >= _SPLIT_DEFICIT_THRESHOLD_US and ag_start.name.startswith(_SPLITTABLE_NAME_PREFIXES):
                    eff_positions = [
                        _effective_producer_pos(op, positions)
                        for op in ag_start.operands()
                    ]
                    split_candidates.append(_SplitCandidate(
                        start_name=ag_start.name,
                        done_name=ag_done.name,
                        deficit_us=deficit,
                        comp_name=comp.name,
                        effective_producer_pos=eff_positions,
                        total_latency_us=collective_latency,
                    ))
                continue

            to_move_set = set(to_move) | {ag_start}
            # All moved items are at positions >= floor (a data/control
            # predecessor can never sit after the instruction it gates), so
            # the insertion index in the filtered new_seq equals floor.
            new_seq = [inst for inst in seq if inst not in to_move_set]
            ins_pos = floor
            for inst in to_move:  # already in topological (schedule) order
                new_seq.insert(ins_pos, inst)
                ins_pos += 1
            new_seq.insert(ins_pos, ag_start)
            schedule.set_sequence(comp, new_seq)
            seq = new_seq
            positions = {inst: i for i, inst in enumerate(seq)}
            name_to_pos = {inst.name: i for inst, i in positions.items()}
            changed = True

            new_ag_start_pos = positions[ag_start]
            new_ag_done_pos = positions[ag_done]
            new_overlap = sum(
                _resolve_inst_cost(seq[i], comp_by_name)
                for i in range(new_ag_start_pos + 1, new_ag_done_pos)
            )
            _logger.info(
                "collective_overlap_pass: moving %s from pos %d to %d "
                "with %d relocated operand(s) (overlap %.1f -> %.1f us, "
                "latency=%.1f us).",
                ag_start.name, orig_ag_start_pos, new_ag_start_pos,
                len(to_move), current_overlap, new_overlap, collective_latency,
            )

            if new_overlap < collective_latency:
                remaining_deficit = collective_latency - new_overlap
                if (
                    remaining_deficit >= _SPLIT_DEFICIT_THRESHOLD_US
                    and ag_start.name.startswith(_SPLITTABLE_NAME_PREFIXES)
                ):
                    eff_positions = [
                        _effective_producer_pos(op, positions)
                        for op in ag_start.operands()
                    ]
                    split_candidates.append(_SplitCandidate(
                        start_name=ag_start.name,
                        done_name=ag_done.name,
                        deficit_us=remaining_deficit,
                        comp_name=comp.name,
                        effective_producer_pos=eff_positions,
                        total_latency_us=collective_latency,
                    ))
            elif profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES):
                done_moved, seq, positions, name_to_pos = _relocate_fsdp_done_before_te_ep(
                    ag_start, ag_done, comp, schedule, seq, positions, name_to_pos,
                    comp_by_name, collective_latency, module_name,
                )
                if done_moved:
                    changed = True

        heavy_changed, seq, positions, name_to_pos = _fill_exposed_collectives_best_of(
            seq, schedule, comp, start_of_done, positions, name_to_pos, comp_by_name, module_name,
            is_while_body=is_while_body,
        )
        if heavy_changed:
            changed = True
            # Some split candidates queued above may now be (partially or
            # fully) hidden by a relocated heavy compute -- drop any whose
            # collective is no longer under-hidden, so phase 2 doesn't pay
            # split overhead for a deficit that no longer exists.
            still_needed = []
            for cand in split_candidates:
                if cand.comp_name != comp.name:
                    still_needed.append(cand)
                    continue
                start_pos = name_to_pos.get(cand.start_name)
                done_pos = name_to_pos.get(cand.done_name)
                if start_pos is None or done_pos is None:
                    still_needed.append(cand)
                    continue
                overlap = sum(
                    _resolve_inst_cost(seq[i], comp_by_name)
                    for i in range(start_pos + 1, done_pos)
                )
                latency = overlap + cand.deficit_us
                remaining = latency - overlap
                if remaining >= _SPLIT_DEFICIT_THRESHOLD_US:
                    cand.deficit_us = remaining
                    still_needed.append(cand)
            split_candidates = still_needed

        if is_while_body and _HOIST_FSDP_STARTS:
            hoist_changed, seq, positions, name_to_pos = _hoist_fsdp_starts_to_floor(
                seq, schedule, comp, start_of_done, positions, name_to_pos,
                comp_by_name, module_name,
            )
            if hoist_changed:
                changed = True

        # DIAGNOSTIC (temporary): re-check every "already hidden" verdict
        # from this comp against the final seq/positions, since neither the
        # per-collective loop nor the fill step above ever revisits it (job
        # 3159101 saw an "already hidden" collective end up outside its
        # window in the final module with no relocation log line to explain
        # it).
        for snap in already_hidden_snapshot:
            ag_start = snap["ag_start"]
            ag_done = snap["ag_done"]
            if ag_start not in positions or ag_done not in positions:
                _logger.warning(
                    "collective_overlap_pass [%s]: DIAG %s / %s dropped "
                    "from positions after fill (was window [%d, %d)).",
                    module_name, ag_start.name, ag_done.name,
                    snap["start_pos"], snap["done_pos"],
                )
                continue
            new_start_pos = positions[ag_start]
            new_done_pos = positions[ag_done]
            new_window_names = [seq[i].name for i in range(new_start_pos + 1, new_done_pos)]
            new_overlap = sum(
                _resolve_inst_cost(seq[i], comp_by_name)
                for i in range(new_start_pos + 1, new_done_pos)
            )
            if (
                new_start_pos != snap["start_pos"]
                or new_done_pos != snap["done_pos"]
                or new_window_names != snap["window_names"]
            ):
                missing = [n for n in snap["window_names"] if n not in new_window_names]
                _logger.warning(
                    "collective_overlap_pass [%s]: DIAG %s window drifted "
                    "after fill: was [%d, %d) contents=%s (overlap=%.1f, "
                    "latency=%.1f) -> now [%d, %d) contents=%s (overlap=%.1f) "
                    "-- vanished from window: %s%s",
                    module_name, ag_start.name,
                    snap["start_pos"], snap["done_pos"], snap["window_names"],
                    snap["overlap"], snap["latency"],
                    new_start_pos, new_done_pos, new_window_names, new_overlap,
                    missing,
                    " *** NO LONGER HIDDEN ***" if new_overlap < snap["latency"] else "",
                )

    return changed, split_candidates


# ---------------------------------------------------------------------------
# Phase 2: split batched collectives at proto level
# ---------------------------------------------------------------------------
def _group_operands_by_epoch(
    effective_positions: list[int],
) -> list[list[int]]:
    """Split operand indices into equal-rank groups ordered by effective position.

    We sort operands by their effective producer position, then divide into
    ceil(n / _SPLIT_GROUP_SIZE) equal-size buckets.  This rank-based approach
    works even when the per-layer GEMMs are back-to-back with no position gap
    (which is what LHS produces to maximise compute throughput).

    Returns a list of ≥2 groups; returns [] if fewer than 2 groups would result.
    """
    n = len(effective_positions)
    n_groups = max(_SPLIT_MIN_GROUPS, (n + _SPLIT_GROUP_SIZE - 1) // _SPLIT_GROUP_SIZE)
    n_groups = min(n_groups, n // 2)  # guarantee ≥2 operands per group

    indexed = sorted(enumerate(effective_positions), key=lambda x: x[1])

    groups: list[list[int]] = []
    base, remainder = divmod(n, n_groups)
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < remainder else 0)
        groups.append([indexed[start + i][0] for i in range(size)])
        start += size

    return groups if len(groups) >= _SPLIT_MIN_GROUPS else []


def _toposort_instructions(insts):
    """Topological sort (Kahn's algorithm) over instruction proto list.

    XLA's CreateFromProto requires each instruction's operands to appear before
    it in the proto.instructions list.  When we add new instructions and reroute
    operand IDs, the original ordering may be violated — this restores a valid
    topological order.  Falls back to the original order on cycles.
    """
    from collections import deque as _deque

    id_set = {i.id for i in insts}
    id_to_inst = {i.id: i for i in insts}

    in_degree: dict[int, int] = {i.id: 0 for i in insts}
    users: dict[int, list] = {i.id: [] for i in insts}

    for inst in insts:
        deps = list(inst.operand_ids) + list(inst.control_predecessor_ids)
        for op_id in deps:
            if op_id in id_set:
                in_degree[inst.id] += 1
                users[op_id].append(inst.id)

    queue = _deque(i.id for i in insts if in_degree[i.id] == 0)
    result = []
    while queue:
        iid = queue.popleft()
        result.append(id_to_inst[iid])
        for uid in users[iid]:
            in_degree[uid] -= 1
            if in_degree[uid] == 0:
                queue.append(uid)

    return result if len(result) == len(insts) else insts


def _phase2_split_core(serialized_hlo: bytes, candidates: list[_SplitCandidate]) -> Optional[bytes]:
    """Proto-level surgery: split each candidate into per-epoch sub-collectives.

    This function imports hlo_pb2 directly and must run in a subprocess to avoid
    protobuf descriptor pool conflicts with jaxlib's pre-registered protos.
    """
    if not candidates:
        return None

    _ensure_protos()
    from xla.service import hlo_pb2  # type: ignore  # pylint: disable=import-outside-toplevel
    from xla import xla_data_pb2  # type: ignore  # pylint: disable=import-outside-toplevel
    _TUPLE = xla_data_pb2.TUPLE  # = 13

    proto = hlo_pb2.HloModuleProto()
    proto.ParseFromString(serialized_hlo)
    sys.stderr.write(f"[split_core] parsed proto: {proto.name}, "
                     f"{len(proto.computations)} computations\n")

    id_to_comp = {c.id: c for c in proto.computations}
    name_to_comp = {c.name: c for c in proto.computations}

    # Find the module-level entry computation (largest scheduled non-fusion).
    module_entry_comp = None
    for c in proto.computations:
        if c.is_fusion_computation:
            continue
        if c.id not in proto.schedule.sequences:
            continue
        if module_entry_comp is None or (
            len(proto.schedule.sequences[c.id].instruction_ids) >
            len(proto.schedule.sequences[module_entry_comp.id].instruction_ids)
        ):
            module_entry_comp = c
    if module_entry_comp is None:
        sys.stderr.write("[split_core] no module entry comp found\n")
        return None

    # Determine which computation to operate on for each candidate.
    # Candidates from while-body computations carry comp_name; fall back to
    # the module entry computation for legacy candidates without comp_name.
    def _target_comp_for(cand):
        if cand.comp_name:
            c = name_to_comp.get(cand.comp_name)
            if c is None:
                sys.stderr.write(
                    f"[split_core] comp_name '{cand.comp_name}' not found, "
                    f"falling back to module entry\n"
                )
                return module_entry_comp
            return c
        return module_entry_comp

    # All candidates must target the same computation per split_core invocation
    # (each subprocess call handles one batch of candidates from one computation).
    entry_comp = _target_comp_for(candidates[0])
    sys.stderr.write(f"[split_core] target comp: {entry_comp.name} "
                     f"({len(entry_comp.instructions)} insts)\n")

    name_to_id = {inst.name: inst.id for inst in entry_comp.instructions}
    id_to_inst = {inst.id: inst for inst in entry_comp.instructions}

    sched_ids = list(proto.schedule.sequences[entry_comp.id].instruction_ids)
    id_to_sched_pos = {iid: pos for pos, iid in enumerate(sched_ids)}

    # ID allocators.
    # Instruction IDs are packed as (computation_unique_id << 32) | local_id.
    # Allocating simply from global_max+1 picks a value whose lower 32 bits
    # may equal an existing local_id in entry_comp -> collision. Instead:
    # new instructions in entry_comp get entry_comp's own parent bits and a
    # local_id above its current max; new async computations use their own
    # comp id as the parent bits, starting at 0.
    _MASK32 = 0xFFFFFFFF
    _entry_parent_bits = entry_comp.id << 32
    _max_entry_local = max(
        (i.id & _MASK32 for i in entry_comp.instructions), default=-1
    )
    _next_entry_local = [_max_entry_local + 1]

    def _new_iid():
        """New instruction ID in entry_comp — same parent bits, fresh local id."""
        v = _next_entry_local[0]; _next_entry_local[0] += 1
        return _entry_parent_bits | (v & _MASK32)

    _next_comp_id = [max((c.id for c in proto.computations), default=0) + 1]

    def _new_cid():
        v = _next_comp_id[0]; _next_comp_id[0] += 1; return v

    # Instructions in NEW async computations get their own parent bits.
    # _new_async_iid(comp_id, local_counter) returns a packed ID for that comp.
    def _new_async_iid(comp_parent_bits: int, local_ctr: list) -> int:
        v = local_ctr[0]; local_ctr[0] += 1
        return comp_parent_bits | (v & _MASK32)

    # Collect all used channel_ids to allocate fresh ones for sub-collectives.
    _used_channels: set[int] = set()
    for _c in proto.computations:
        for _i in _c.instructions:
            if _i.channel_id:
                _used_channels.add(_i.channel_id)
    _next_channel_id = [max(_used_channels, default=0) + 1]

    def _new_channel():
        v = _next_channel_id[0]; _next_channel_id[0] += 1; return v

    any_split = False

    for cand in candidates:
        sys.stderr.write(f"[split_core] processing candidate: {cand.start_name} "
                         f"(deficit={cand.deficit_us:.1f}us, "
                         f"n_ep={len(cand.effective_producer_pos)} operands)\n")
        start_id = name_to_id.get(cand.start_name)
        if start_id is None:
            sys.stderr.write(f"[split_core] start {cand.start_name!r} not in proto\n")
            sys.stderr.write(f"[split_core] known names sample: "
                             f"{list(name_to_id)[:5]}\n")
            continue
        start_inst = id_to_inst.get(start_id)
        if start_inst is None:
            sys.stderr.write(f"[split_core] start_id {start_id} not in id_to_inst\n")
            continue

        # Find the async-done (its only operand is start_inst)
        done_inst = None
        done_id_search = name_to_id.get(cand.done_name)
        if done_id_search:
            done_inst = id_to_inst.get(done_id_search)
        if done_inst is None:
            sys.stderr.write(f"[split_core] done {cand.done_name!r} not in proto\n")
            continue

        n_operands = len(start_inst.operand_ids)
        sys.stderr.write(f"[split_core] {cand.start_name}: n_operands={n_operands}\n")
        if n_operands < 2:
            sys.stderr.write(f"[split_core] {cand.start_name}: too few operands, skip\n")
            continue

        if not start_inst.called_computation_ids:
            sys.stderr.write(f"[split_core] {cand.start_name}: no called_computation_ids\n")
            continue
        called_comp = id_to_comp.get(start_inst.called_computation_ids[0])
        if called_comp is None:
            sys.stderr.write(f"[split_core] {cand.start_name}: called comp not found\n")
            continue

        # Find inner collective op and params in called computation
        inner_inst = None
        param_map: dict[int, object] = {}
        for inst in called_comp.instructions:
            if inst.opcode in ("reduce-scatter", "all-reduce", "all-gather",
                               "collective-permute"):
                inner_inst = inst
            elif inst.opcode == "parameter":
                param_map[inst.parameter_number] = inst
        if inner_inst is None:
            sys.stderr.write(
                f"[split_core] no inner collective in {cand.start_name} called comp "
                f"(opcodes: {[i.opcode for i in called_comp.instructions]})\n"
            )
            continue

        # Group operands by effective producer epoch.
        # Note: cand.effective_producer_pos comes from phase-1 C++ schedule
        # positions which may differ from the proto schedule positions.  We
        # use them only for grouping (relative order is preserved); the actual
        # insertion positions are recomputed via _proto_effective_pos below.
        groups = _group_operands_by_epoch(cand.effective_producer_pos)
        _sorted_pos = sorted(cand.effective_producer_pos)
        _gaps = [_sorted_pos[i+1] - _sorted_pos[i] for i in range(len(_sorted_pos)-1)]
        _max_gap = max(_gaps) if _gaps else 0
        sys.stderr.write(
            f"[split_core] {cand.start_name}: ep_pos min={min(cand.effective_producer_pos)} "
            f"max={max(cand.effective_producer_pos)} "
            f"span={max(cand.effective_producer_pos)-min(cand.effective_producer_pos)}, "
            f"max_consecutive_gap={_max_gap}, groups={len(groups)}\n"
        )
        sys.stderr.write(f"[split_core] sorted_ep_pos (phase1, may be stale): {_sorted_pos}\n")
        # Log fresh proto positions for debugging.
        # depth=8: traces to GEMM; depth=1: stops at direct dep (GTE for bitcasts).
        _fresh_ep8 = [
            _proto_effective_pos(start_inst.operand_ids[i], id_to_inst, id_to_sched_pos, max_depth=8)
            for i in range(len(start_inst.operand_ids))
        ]
        _fresh_ep1 = [
            _proto_effective_pos(start_inst.operand_ids[i], id_to_inst, id_to_sched_pos, max_depth=1)
            for i in range(len(start_inst.operand_ids))
        ]
        sys.stderr.write(f"[split_core] sorted_ep_pos (proto, depth=8): {sorted(_fresh_ep8)}\n")
        sys.stderr.write(f"[split_core] sorted_ep_pos (proto, depth=1): {sorted(_fresh_ep1)}\n")
        if not groups:
            sys.stderr.write(
                f"[split_core] {cand.start_name}: could not form ≥2 groups "
                f"(n={len(cand.effective_producer_pos)}, group_size={_SPLIT_GROUP_SIZE}), "
                f"cannot split\n"
            )
            continue

        # Per-operand byte sizes, used below to estimate each new
        # sub-collective's own latency as its share of the original
        # collective's PGLE-measured total_latency_us, weighted by bytes
        # transferred rather than splitting evenly across groups. Without
        # this, the post-split _phase1_reorder re-run has no PGLE entry for
        # any new sub-collective and silently skips them -- see
        # _SplitCandidate.total_latency_us.
        #
        # A byte-share-based "skip the split if a group looks too small"
        # heuristic was tried to guard against the downstream-consumer
        # imbalance that starved all-gather-start.8.g0/.g3 (jobs
        # 3191924/3193061) and was reverted: byte share doesn't predict
        # downstream consumer count -- g2 (7 dependent GEMMs) had the
        # *smallest* byte share (0.19x even split) while g0 (1 dependent
        # GEMM, the actual problem) had a roughly-average share (0.91x).
        _operand_bytes = [
            _proto_shape_bytes(id_to_inst[start_inst.operand_ids[i]].shape)
            for i in range(n_operands)
        ]
        _total_operand_bytes = sum(_operand_bytes) or 1

        epoch_summaries = [
            f"g{i}:{len(g)}ops@pos{max(cand.effective_producer_pos[j] for j in g)}"
            for i, g in enumerate(groups)
        ]
        _logger.info(
            "collective_overlap_pass: splitting %s (deficit=%.1f us) "
            "into %d groups: %s",
            cand.start_name, cand.deficit_us, len(groups), epoch_summaries,
        )

        # --- For each group, build a sub-collective ---
        # new_pairs: (group_indices, new_start_id, new_done_id,
        # (group_indices, new_start_id, new_done_id, effective_insert_after_pos, group_op_ids)
        new_pairs: list[tuple[list[int], int, int, int, list[int]]] = []
        _seen_gte_ids: set[int] = set()  # guards against duplicate intermediate insertion
        _ZERO_COST_OPS = frozenset(("bitcast", "get-tuple-element", "tuple"))

        for g_idx, group in enumerate(groups):
            effective_insert_after = max(
                _proto_effective_pos(start_inst.operand_ids[i], id_to_inst, id_to_sched_pos)
                for i in group
            )
            group_op_ids: list[int] = []
            for i in group:
                op_id = start_inst.operand_ids[i]
                intermediates: list[int] = []
                cur_id = op_id
                while True:
                    inst = id_to_inst.get(cur_id)
                    if (inst is None or inst.opcode not in _ZERO_COST_OPS
                            or not inst.operand_ids):
                        break
                    next_id = inst.operand_ids[0]
                    next_inst = id_to_inst.get(next_id)
                    if next_inst is None or next_inst.opcode not in _ZERO_COST_OPS:
                        break
                    if next_id not in _seen_gte_ids:
                        intermediates.append(next_id)
                        _seen_gte_ids.add(next_id)
                    cur_id = next_id
                group_op_ids.extend(reversed(intermediates))
                # op_id itself only needs relocating when it is a zero-cost
                # op (bitcast/GTE/tuple) directly feeding the
                # collective-start -- it must move with the rest of the
                # chain so the sub-start's operands stay contiguous. If
                # op_id is the real (non-zero-cost) producer instead (e.g. a
                # GEMM with no zero-cost wrapper), it must NOT be relocated:
                # moving heavy compute can violate other consumers'
                # ordering, and removing op_id from the schedule invalidates
                # its cached position in orig_to_compact, which silently
                # falls back to the raw pre-removal index -- overshooting
                # the true compact position badly enough that the sub-done
                # can end up inserted before its own sub-start (RET_CHECK at
                # hlo_schedule.cc:456).
                #
                # _seen_gte_ids dedup still applies: two operand indices can
                # reference the same zero-cost instruction, or one operand's
                # op_id can turn out to be an ancestor discovered while
                # walking a later operand's chain -- either way it must
                # only be relocated once, or XLA's schedule verifier trips
                # on the duplicate insertion (hlo_schedule.cc:439).
                _op_inst = id_to_inst.get(op_id)
                if (_op_inst is not None and _op_inst.opcode in _ZERO_COST_OPS
                        and op_id not in _seen_gte_ids):
                    group_op_ids.append(op_id)
                    _seen_gte_ids.add(op_id)

            # New async computation
            new_cid = _new_cid()
            nc = proto.computations.add()
            nc.id = new_cid
            nc.name = f"{called_comp.name}.g{g_idx}"
            nc.is_fusion_computation = False
            nc.execution_thread = called_comp.execution_thread

            # Parameters — use the new computation's parent bits for its instructions
            _nc_parent_bits = new_cid << 32
            _nc_local_ctr = [0]
            new_param_ids: list[int] = []
            for new_idx, orig_idx in enumerate(group):
                orig_p = param_map[orig_idx]
                pid = _new_async_iid(_nc_parent_bits, _nc_local_ctr)
                p = nc.instructions.add()
                p.id = pid
                p.name = f"param_{new_idx}.{nc.name}"
                p.opcode = "parameter"
                p.parameter_number = new_idx
                p.shape.CopyFrom(orig_p.shape)
                new_param_ids.append(pid)

            # Inner collective (same opcode / dims / replica_groups as original)
            new_rs_id = _new_async_iid(_nc_parent_bits, _nc_local_ctr)
            nr = nc.instructions.add()
            nr.id = new_rs_id
            nr.name = f"{inner_inst.name}.g{g_idx}"
            nr.opcode = inner_inst.opcode
            nr.operand_ids.extend(new_param_ids)

            _group_bytes = sum(_operand_bytes[i] for i in group)
            _group_latency_us = cand.total_latency_us * (_group_bytes / _total_operand_bytes)
            sys.stderr.write(
                f"[split_core] GROUP_LATENCY {nr.name} {_group_latency_us:.6f} "
                f"(bytes={_group_bytes}/{_total_operand_bytes}, "
                f"total_latency={cand.total_latency_us:.1f}us)\n"
            )
            nr.dimensions.extend(inner_inst.dimensions)
            # Copy the replica-group spec verbatim. Modern XLA usually encodes
            # this via collective_device_list (or iota_collective_device_list)
            # rather than the legacy replica_groups field, which is then left
            # empty — copying only replica_groups silently drops the real
            # group, so the verifier falls back to inferring a full-device
            # subgroup (e.g. 32) instead of the true, possibly smaller,
            # subgroup (e.g. 8), tripping the shard_count == subgroup_size
            # RET_CHECK in hlo_verifier.cc.
            nr.replica_groups.extend(inner_inst.replica_groups)
            # The modern replacement is the "replica_group_list" oneof
            # (collective_device_list / iota_collective_device_list /
            # mesh_axes_replica_group_list, the latter for Shardy-partitioned
            # modules). Whichever variant is set, the device grouping is
            # identical across all split sub-collectives (splitting only
            # partitions the operand/buffer list, not who talks to whom), so
            # copy it verbatim.
            _which_dl = inner_inst.WhichOneof("replica_group_list")
            if _which_dl is not None:
                getattr(nr, _which_dl).CopyFrom(getattr(inner_inst, _which_dl))
            nr.use_global_device_ids = inner_inst.use_global_device_ids
            # collective-permute doesn't use replica_groups/device_list at all
            # (HloCollectivePermuteInstruction extends HloChannelInstruction,
            # not HloCollectiveInstruction) — its participants are defined
            # entirely by source_target_pairs, which none of the copies above
            # touch. Without this, a split collective-permute would silently
            # get an empty pairing.
            if inner_inst.opcode == "collective-permute":
                nr.source_target_pairs.extend(inner_inst.source_target_pairs)
            # Copy the to_apply reduction computation (e.g. add.47.clone)
            nr.called_computation_ids.extend(inner_inst.called_computation_ids)
            if inner_inst.channel_id:
                nr.channel_id = _new_channel()  # must be unique per collective
            nr.metadata.CopyFrom(inner_inst.metadata)
            if inner_inst.backend_config:
                nr.backend_config = inner_inst.backend_config
            # Output shape: tuple of the subset of the original output tuple elements
            nr.shape.element_type = _TUPLE
            for orig_idx in group:
                s = nr.shape.tuple_shapes.add()
                s.CopyFrom(inner_inst.shape.tuple_shapes[orig_idx])
            nc.root_id = new_rs_id

            # Schedule sequence for the new async computation — XLA requires every
            # non-fusion computation to have an entry in proto.schedule.sequences.
            nc_seq = proto.schedule.sequences[new_cid]
            for _p in nc.instructions:
                nc_seq.instruction_ids.append(_p.id)

            # async-start in main computation
            ns_id = _new_iid()
            ns = entry_comp.instructions.add()
            ns.id = ns_id
            ns.name = f"{cand.start_name}.g{g_idx}"
            ns.opcode = "async-start"
            ns.async_execution_thread = start_inst.async_execution_thread
            ns.called_computation_ids.append(new_cid)
            for orig_idx in group:
                ns.operand_ids.append(start_inst.operand_ids[orig_idx])
            ns.metadata.CopyFrom(start_inst.metadata)
            if start_inst.backend_config:
                ns.backend_config = start_inst.backend_config
            if start_inst.frontend_attributes.map:
                ns.frontend_attributes.CopyFrom(start_inst.frontend_attributes)
            # Shape: (context_tuple, output_tuple) — both sub-tuples need TUPLE type
            ns.shape.element_type = _TUPLE
            ctx = ns.shape.tuple_shapes.add()
            ctx.element_type = _TUPLE
            for orig_idx in group:
                s = ctx.tuple_shapes.add()
                s.CopyFrom(start_inst.shape.tuple_shapes[0].tuple_shapes[orig_idx])
            out = ns.shape.tuple_shapes.add()
            out.element_type = _TUPLE
            for orig_idx in group:
                s = out.tuple_shapes.add()
                s.CopyFrom(start_inst.shape.tuple_shapes[1].tuple_shapes[orig_idx])

            # async-done in main computation
            nd_id = _new_iid()
            nd = entry_comp.instructions.add()
            nd.id = nd_id
            nd.name = f"{cand.done_name}.g{g_idx}"
            nd.opcode = "async-done"
            nd.operand_ids.append(ns_id)
            nd.metadata.CopyFrom(done_inst.metadata)
            if done_inst.backend_config:
                nd.backend_config = done_inst.backend_config
            if done_inst.frontend_attributes.map:
                nd.frontend_attributes.CopyFrom(done_inst.frontend_attributes)
            # Shape: output tuple (subset of original done output elements)
            nd.shape.element_type = _TUPLE
            for orig_idx in group:
                s = nd.shape.tuple_shapes.add()
                s.CopyFrom(start_inst.shape.tuple_shapes[1].tuple_shapes[orig_idx])

            new_pairs.append((group, ns_id, nd_id, effective_insert_after, group_op_ids))

        # --- Reroute GTE users of old done to appropriate split done ---
        orig_to_new: dict[int, tuple[int, int]] = {}
        for g_idx, (group, _, nd_id, _, _) in enumerate(new_pairs):
            for new_idx, orig_idx in enumerate(group):
                orig_to_new[orig_idx] = (nd_id, new_idx)

        for inst in entry_comp.instructions:
            if (inst.opcode == "get-tuple-element" and
                    len(inst.operand_ids) == 1 and
                    inst.operand_ids[0] == done_inst.id):
                orig_idx = inst.tuple_index
                if orig_idx in orig_to_new:
                    new_nd_id, new_idx = orig_to_new[orig_idx]
                    inst.operand_ids[0] = new_nd_id
                    inst.tuple_index = new_idx

        old_done_sched_pos = id_to_sched_pos.get(done_inst.id, len(sched_ids))

        sys.stderr.write(f"[split_core] SCHED_POS {cand.start_name}: "
                         f"{id_to_sched_pos.get(start_inst.id, -1)}\n")
        sys.stderr.write(f"[split_core] SCHED_POS {cand.done_name}: "
                         f"{old_done_sched_pos}\n")

        # All direct operand IDs being repositioned (zero-cost bitcasts/GTEs).
        _all_group_op_ids: set[int] = set()
        for (_, _, _, _, op_ids) in new_pairs:
            _all_group_op_ids.update(op_ids)

        for (grp, ns_id, nd_id, eff_pos, op_ids) in new_pairs:
            sys.stderr.write(f"[split_core] GROUP eff_pos={eff_pos}: "
                             f"op_ids sched_pos={sorted(id_to_sched_pos.get(oid, -1) for oid in op_ids)}\n")

        _remove_from_sched = {start_inst.id, done_inst.id} | _all_group_op_ids
        new_sched = [iid for iid in sched_ids if iid not in _remove_from_sched]

        # Map original positions → compact positions (after all removals).
        orig_to_compact: dict[int, int] = {}
        _cidx = 0
        for _oi, _iid in enumerate(sched_ids):
            if _iid not in _remove_from_sched:
                orig_to_compact[_oi] = _cidx
                _cidx += 1

        def _compact_pos_at_or_before(pos: int) -> int:
            # eff_pos should land on a surviving instruction, but as
            # defense-in-depth, snap to the nearest surviving position at
            # or before it rather than the raw pre-removal index -- the raw
            # index can overshoot into compact positions reserved for later
            # insertions (e.g. the sub-done block) and trip the RET_CHECK
            # ordering violation at hlo_schedule.cc:456.
            for _p in range(pos, -1, -1):
                if _p in orig_to_compact:
                    return orig_to_compact[_p]
            return 0

        # Insert each group's bitcasts + sub-start right after its GEMM.
        sorted_pairs = sorted(new_pairs, key=lambda x: x[3])

        for group, ns_id, nd_id, eff_pos, op_ids in sorted_pairs:
            _cpct = _compact_pos_at_or_before(eff_pos)
            sys.stderr.write(f"[split_core] GROUP eff_pos={eff_pos} -> compact={_cpct}\n")
        offset = 0
        for group, ns_id, nd_id, eff_pos, op_ids in sorted_pairs:
            compact_pos = _compact_pos_at_or_before(eff_pos)
            ins_pos = compact_pos + 1 + offset
            # Reinsert the zero-cost operands first, then the sub-start.
            for _op_id in op_ids:
                new_sched.insert(ins_pos, _op_id)
                ins_pos += 1
                offset += 1
            new_sched.insert(ins_pos, ns_id)
            offset += 1

        # Insert sub-dones just before the first surviving instruction after
        # where the original done was.
        compact_done_pos = len(new_sched)
        for _i in range(old_done_sched_pos + 1, len(sched_ids)):
            if _i in orig_to_compact:
                compact_done_pos = orig_to_compact[_i]
                break
        done_insert_base = compact_done_pos + offset
        for g_idx, (_, _, nd_id, _, _) in enumerate(sorted_pairs):
            new_sched.insert(done_insert_base + g_idx, nd_id)

        del proto.schedule.sequences[entry_comp.id].instruction_ids[:]
        proto.schedule.sequences[entry_comp.id].instruction_ids.extend(new_sched)
        sched_ids = new_sched
        id_to_sched_pos = {iid: pos for pos, iid in enumerate(sched_ids)}

        _remove_ids = {start_inst.id, done_inst.id}
        _surviving = [i for i in entry_comp.instructions if i.id not in _remove_ids]
        _ordered = _toposort_instructions(_surviving)
        del entry_comp.instructions[:]
        for _inst in _ordered:
            entry_comp.instructions.add().CopyFrom(_inst)

        # Rebuild lookup maps for subsequent candidates
        id_to_inst = {inst.id: inst for inst in entry_comp.instructions}
        name_to_id = {inst.name: inst.id for inst in entry_comp.instructions}

        any_split = True
        _logger.info(
            "collective_overlap_pass: split %s into %d sub-collectives (%s)",
            cand.start_name, len(groups), epoch_summaries,
        )

    if not any_split:
        return None

    # XLA's CreateFromProto processes computations in proto order and builds
    # the computation_map incrementally -- every callee must appear before
    # its caller. A "module entry must be last" special case is NOT enough:
    # entry_comp (the split target for this invocation, see _target_comp_for
    # above) can itself be a while-body computation rather than the true
    # module entry, and it gains new callees too (the newly created async
    # sub-computations) -- but as a pre-existing non-entry computation it
    # would otherwise keep its original, earlier position, landing before
    # its own new callees, which are simply appended at the end. Confirmed
    # failing in practice (job 3203819): "all-gather-start.6.g0 instruction
    # references invalid computation id(s)" (RET_CHECK at
    # hlo_instruction.cc:391) from exactly this case. Full topological sort
    # by actual call graph (post-order DFS, callees appended before their
    # caller) handles both the true-entry and while-body-target cases
    # uniformly.
    _id_to_proto_comp = {c.id: c for c in proto.computations}
    _visited: set[int] = set()
    _topo_ordered: list = []

    def _visit_comp(_cid: int) -> None:
        if _cid in _visited:
            return
        _visited.add(_cid)
        _c = _id_to_proto_comp.get(_cid)
        if _c is None:
            return
        for _inst in _c.instructions:
            for _callee_id in _inst.called_computation_ids:
                _visit_comp(_callee_id)
        _topo_ordered.append(_c)

    for _c in proto.computations:
        _visit_comp(_c.id)
    del proto.computations[:]
    for _c in _topo_ordered:
        proto.computations.add().CopyFrom(_c)
    sys.stderr.write(
        f"[split_core] topologically reordered computations: "
        f"{len(_topo_ordered)} total\n"
    )

    return proto.SerializeToString()


_GROUP_LATENCY_RE = re.compile(r"^\[split_core\] GROUP_LATENCY (\S+) ([0-9.eE+-]+)")


def _phase2_split_one_comp(
    serialized_hlo: bytes, candidates: list[_SplitCandidate]
) -> Optional[tuple[bytes, dict[str, float]]]:
    """Spawn a subprocess to run _phase2_split_core for one computation's
    worth of candidates.

    _phase2_split_core resolves its target computation from
    candidates[0].comp_name and looks every other candidate up by name in
    that same computation -- callers (_phase2_split) must pre-group
    candidates by comp_name and call this once per group, since a single
    invocation silently drops any candidate that doesn't belong to
    candidates[0]'s computation.

    The subprocess avoids protobuf descriptor pool conflicts that arise when
    jaxlib pre-registers xla/service/metrics.proto in the host process.
    """
    if not candidates:
        return None

    import json as _json
    import tempfile as _tempfile

    with _tempfile.NamedTemporaryFile(delete=False, suffix=".hlo.bin") as _tf:
        _tf.write(serialized_hlo)
        _hlo_file = _tf.name

    try:
        _candidates_json = _json.dumps([{
            "start_name": c.start_name,
            "done_name": c.done_name,
            "deficit_us": c.deficit_us,
            "comp_name": c.comp_name,
            "effective_producer_pos": c.effective_producer_pos,
            "total_latency_us": c.total_latency_us,
        } for c in candidates])

        result = subprocess.run(
            [sys.executable, __file__, "--split", _hlo_file, _candidates_json],
            capture_output=True,
            timeout=120,
        )

        if result.returncode != 0:
            _logger.warning(
                "collective_overlap_pass: split subprocess failed (rc=%d):\n%s",
                result.returncode,
                result.stderr.decode("utf-8", errors="replace")[-3000:],
            )
            return None

        # Always log subprocess stderr for diagnostics.
        _sub_stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if _sub_stderr:
            _logger.info(
                "collective_overlap_pass: split subprocess stderr:\n%s", _sub_stderr
            )

        # Per-sub-collective latency estimates (profile_key -> latency_us),
        # emitted by _phase2_split_core since the new sub-collectives have
        # no PGLE profile entry of their own -- see
        # _SplitCandidate.total_latency_us.
        group_latencies: dict[str, float] = {}
        for _line in _sub_stderr.splitlines():
            _m = _GROUP_LATENCY_RE.match(_line)
            if _m:
                group_latencies[_m.group(1)] = float(_m.group(2))

        if result.stdout:
            _logger.info(
                "collective_overlap_pass: split subprocess succeeded (%d bytes, "
                "%d group latency estimate(s)).",
                len(result.stdout), len(group_latencies),
            )
            return result.stdout, group_latencies

        _logger.info("collective_overlap_pass: split subprocess produced no output.")
        return None

    finally:
        try:
            os.unlink(_hlo_file)
        except Exception:
            pass


def _phase2_split(
    serialized_hlo: bytes, candidates: list[_SplitCandidate]
) -> Optional[tuple[bytes, dict[str, float]]]:
    """Split every candidate, across however many computations they span.

    split_candidates accumulated in _phase1_reorder commonly span multiple
    computations at once (e.g. a while-loop body's call-start wrappers
    alongside the entry computation's own plain all-gather-start), but
    _phase2_split_one_comp / _phase2_split_core only ever look at one
    computation per invocation (resolved from candidates[0].comp_name).
    Group by comp_name and run one subprocess invocation per group,
    chaining each group's output bytes into the next group's input so
    earlier groups' splits are preserved. Returns None only if every group
    failed to produce a split; otherwise returns the fully accumulated
    result even if some groups failed.

    Also merges each group's new-sub-collective latency estimates (see
    _phase2_split_one_comp / _SplitCandidate.total_latency_us) into a single
    dict the caller should feed into _profile_costs before re-running phase
    1 on the split result -- otherwise the new sub-collectives have no PGLE
    entry and the re-run silently skips every one of them.
    """
    if not candidates:
        return None

    groups: dict[str, list[_SplitCandidate]] = {}
    for c in candidates:
        groups.setdefault(c.comp_name or "", []).append(c)

    current_bytes = serialized_hlo
    any_succeeded = False
    all_group_latencies: dict[str, float] = {}
    for comp_name, group in groups.items():
        result = _phase2_split_one_comp(current_bytes, group)
        if result is not None:
            current_bytes, group_latencies = result
            all_group_latencies.update(group_latencies)
            any_succeeded = True
        else:
            _logger.info(
                "collective_overlap_pass: split produced no output for "
                "comp '%s' (%d candidate(s)); leaving them unsplit.",
                comp_name or "<module entry>", len(group),
            )

    return (current_bytes, all_group_latencies) if any_succeeded else None


# ---------------------------------------------------------------------------
# Top-level POST_SCHEDULER pass entry point
# ---------------------------------------------------------------------------
def _diag_recheck_hidden_windows(schedule, module_name: str, label: str) -> None:
    """DIAGNOSTIC (temporary): re-verify every "already hidden" verdict
    recorded in _diag_hidden_windows against a *fresh* read of
    schedule.sequence(comp) -- independent of whatever seq/positions
    _phase1_reorder was tracking internally, to rule out a bug in that
    bookkeeping itself vs. a real schedule mutation between passes.
    """
    for snap in _diag_hidden_windows:
        comp = snap["comp"]
        ag_start = snap["ag_start"]
        ag_done = snap["ag_done"]
        seq = list(schedule.sequence(comp))
        try:
            new_start_pos = seq.index(ag_start)
            new_done_pos = seq.index(ag_done)
        except ValueError:
            _logger.warning(
                "collective_overlap_pass [%s]: DIAG[%s] %s / %s no longer "
                "found in schedule.sequence(comp) at all (was window "
                "[%d, %d)).",
                module_name, label, ag_start.name, ag_done.name,
                snap["start_pos"], snap["done_pos"],
            )
            continue
        new_window_names = [seq[i].name for i in range(new_start_pos + 1, new_done_pos)]
        if (
            new_start_pos != snap["start_pos"]
            or new_done_pos != snap["done_pos"]
            or new_window_names != snap["window_names"]
        ):
            missing = [n for n in snap["window_names"] if n not in new_window_names]
            _logger.warning(
                "collective_overlap_pass [%s]: DIAG[%s] %s window drifted: "
                "was [%d, %d) contents=%s (overlap=%.1f, latency=%.1f) -> "
                "now [%d, %d) contents=%s -- vanished: %s",
                module_name, label, ag_start.name,
                snap["start_pos"], snap["done_pos"], snap["window_names"],
                snap["overlap"], snap["latency"],
                new_start_pos, new_done_pos, new_window_names, missing,
            )
        else:
            _logger.info(
                "collective_overlap_pass [%s]: DIAG[%s] %s window unchanged "
                "([%d, %d), contents=%s) -- still consistent.",
                module_name, label, ag_start.name,
                new_start_pos, new_done_pos, new_window_names,
            )


# Cap on how many times we re-run _phase1_reorder against a freshly
# schedule.update()'d module (see _run_phase1_to_fixed_point's docstring).
# Normal runs converge in 1-2 iterations; this is a safety net against a
# pathological cascade never settling.
_MAX_PHASE1_FIXED_POINT_ITERS = 4


def _log_final_exposed_summary(module, schedule, module_name: str, label: str) -> None:
    """Log the true total exposed time across every computation in `module`,
    computed directly from `schedule`'s live state -- a ground-truth
    measurement independent of intermediate _total_exposed_us values logged
    during earlier steps, which can go stale after a later
    schedule.update()/verify() (see _run_phase1_to_fixed_point's
    docstring). Call immediately before serializing/returning the final
    module.
    """
    comp_by_name = {c.name: c for c in module.computations()}
    grand_total = 0.0
    per_comp = []
    per_collective = []  # (comp_name, collective_name, deficit_us)
    for comp in module.make_nonfusion_computations():
        seq = schedule.sequence(comp)
        if seq is None:
            continue
        seq = list(seq)
        positions = {inst: i for i, inst in enumerate(seq)}
        start_of_done: dict = {}
        for inst in seq:
            if _is_async_start(inst):
                users = list(inst.users())
                if len(users) == 1:
                    start_of_done[users[0]] = inst
        if not start_of_done:
            continue
        total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
        if total > 0:
            per_comp.append((comp.name, total))
        grand_total += total

        prefix = _prefix_costs_excluding(seq, (), comp_by_name)
        for ag_done, ag_start in start_of_done.items():
            if ag_start not in positions or ag_done not in positions:
                continue
            profile_key = _resolve_profile_key(ag_start, comp_by_name)
            latency = _profile_costs.get(profile_key)
            if latency is None or latency <= 0:
                continue
            s, d = positions[ag_start], positions[ag_done]
            overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
            deficit = max(0.0, latency - overlap)
            if deficit > 0:
                per_collective.append((comp.name, ag_start.name, deficit))
    per_comp.sort(key=lambda x: -x[1])
    per_collective.sort(key=lambda x: -x[2])
    _logger.info(
        "collective_overlap_pass [%s]: FINAL ground-truth exposed summary "
        "[%s] (direct from schedule, right before serialization): "
        "grand_total=%.1f us across %d computation(s) with nonzero "
        "exposure: %s",
        module_name, label, grand_total, len(per_comp),
        ", ".join(f"{name}={total:.1f}us" for name, total in per_comp),
    )
    _logger.info(
        "collective_overlap_pass [%s]: FINAL ground-truth per-collective "
        "exposed breakdown [%s]: %s",
        module_name, label,
        ", ".join(
            f"{comp_name}/{name}={deficit:.1f}us"
            for comp_name, name, deficit in per_collective
        ),
    )


def _run_phase1_to_fixed_point(module, schedule, module_name: str) -> tuple[bool, list[_SplitCandidate]]:
    """Run _phase1_reorder to a fixed point against `module`/`schedule`.

    schedule.update()/verify() (required so XLA can canonicalize the
    schedule after our raw seq mutations) can itself silently relocate an
    instruction whose true data dependency one of our own earlier
    relocations invalidated elsewhere in the same comp -- e.g. moving
    collective A can shift instruction X across collective B's done,
    without X itself ever being touched by name. That relocation is
    invisible to _phase1_reorder's own seq/positions bookkeeping, so a
    window scored "already hidden" can end up genuinely exposed by the
    time schedule.update() finishes. Confirmed on job 3161055:
    all-gather-start.9's window was intact immediately after
    _phase1_reorder returned, and only vanished after
    schedule.update()/verify()/set_schedule(), because te_gemm_v2_ffi.93
    has a real data dependency on all-gather-done.8 that got invalidated in
    the same pass. Re-running _phase1_reorder against the *post-update*
    schedule lets it notice the now-genuinely-exposed window and re-fill
    it.

    Also used to re-optimize a module straight out of phase 2 (see
    _compute_collective_overlap): _phase2_split_core is pure proto surgery
    (grouping an existing collective's operands into new, smaller
    sub-collectives + a topological-order fixup) -- it never re-derives
    positions against real PGLE costs or runs the relocate/fill logic, so
    the newly created sub-collectives start out wherever the topological
    fixup happened to place them, not at their own earliest legal position.
    """
    global _fill_strategy_cache
    _fill_strategy_cache = {}
    changed = False
    split_candidates: list[_SplitCandidate] = []
    for _fp_iter in range(_MAX_PHASE1_FIXED_POINT_ITERS):
        iter_changed, split_candidates = _phase1_reorder(module, schedule, module_name)
        _diag_recheck_hidden_windows(
            schedule, module_name,
            f"fixed-point iter {_fp_iter}, immediately after _phase1_reorder return",
        )
        if not iter_changed:
            break
        changed = True
        schedule.update()
        schedule.verify()
        module.set_schedule(schedule)
        _diag_recheck_hidden_windows(
            schedule, module_name,
            f"fixed-point iter {_fp_iter}, after schedule.update()/verify()/set_schedule()",
        )
    else:
        # Every completed iteration above ends with schedule.update()/
        # verify()/set_schedule(), which can itself silently disturb
        # dependencies and re-expose windows _phase1_reorder just finished
        # filling (per this function's docstring). Normally the next
        # iteration's _phase1_reorder call notices and repairs that -- but
        # when the loop exhausts here instead of breaking, the most recent
        # update()/verify() is never rechecked, so its disturbance goes
        # unaddressed. Confirmed in job 3188746: the fill step's own
        # measured ENTRY total converged to ~59750-59770us, but the actual
        # final schedule measured ~155000us exposed across the module --
        # run one more _phase1_reorder pass so the function never returns a
        # schedule whose last mutation was an unverified update().
        _logger.warning(
            "collective_overlap_pass [%s]: phase 1 fixed point not reached "
            "after %d iterations; running one final _phase1_reorder pass "
            "against the post-update schedule so it doesn't go unaddressed.",
            module_name, _MAX_PHASE1_FIXED_POINT_ITERS,
        )
        final_changed, split_candidates = _phase1_reorder(module, schedule, module_name)
        if final_changed:
            changed = True
            # _phase1_reorder only mutates `schedule` directly via
            # set_sequence -- it never itself calls
            # update()/verify()/module.set_schedule() (that's this loop's
            # job, same as every earlier iteration above). Without this,
            # this final pass's relocations would never actually land in
            # the module this function's caller serializes.
            schedule.update()
            schedule.verify()
            module.set_schedule(schedule)
        _diag_recheck_hidden_windows(
            schedule, module_name,
            "post-fixed-point-cap final _phase1_reorder pass",
        )
    return changed, split_candidates


def _compute_collective_overlap(serialized_hlo: bytes) -> Optional[bytes]:
    """Phase 1: reorder; Phase 2: split batched collectives.

    This is the actual (rank-local) computation. It is only ever invoked on
    rank 0 -- see _collective_overlap_pass below, which broadcasts rank 0's
    *output* to every other rank instead of having each rank compute its own
    (potentially divergent) answer.
    """
    if not _profile_costs:
        return None

    from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel
    module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
    schedule = module.schedule()
    if schedule is None:
        return None

    module_name = module.name

    if os.environ.get("COLLECTIVE_OVERLAP_DUMP_MODULE", "1") == "1":
        sys.stderr.write(
            f"[collective_overlap_pass] === BEGIN ORIGINAL MODULE: {module_name} ===\n"
            f"{module.to_string()}\n"
            f"[collective_overlap_pass] === END ORIGINAL MODULE: {module_name} ===\n"
        )

    _log_profile_coverage_gaps(module, schedule, module_name)

    # ---- Phase 1 ----
    changed, split_candidates = _run_phase1_to_fixed_point(module, schedule, module_name)
    _log_final_exposed_summary(module, schedule, module_name, "post-phase-1, pre-phase-2")

    if changed:
        phase1_bytes = module.as_serialized_hlo_module_proto()
    else:
        phase1_bytes = serialized_hlo

    # ---- Phase 2 ----
    if split_candidates:
        _logger.info(
            "collective_overlap_pass [%s]: %d split candidate(s) after phase 1.",
            module_name, len(split_candidates),
        )
        phase2_result = _phase2_split(phase1_bytes, split_candidates)
        if phase2_result is not None:
            phase2_bytes, group_latencies = phase2_result
            # _phase2_split_core is pure proto surgery -- it never reorders
            # or fills the newly created sub-collectives against real PGLE
            # costs, and those sub-collectives have no PGLE entry of their
            # own (they didn't exist when profiling ran). Seed
            # _profile_costs with the byte-weighted latency estimates
            # _phase2_split computed for them, so the post-split phase-1
            # re-run below doesn't just silently skip every one of them.
            if group_latencies:
                _logger.info(
                    "collective_overlap_pass [%s]: seeding %d sub-collective "
                    "latency estimate(s) into _profile_costs before "
                    "re-running phase 1 on the split result.",
                    module_name, len(group_latencies),
                )
                _profile_costs.update(group_latencies)
            # Re-run phase 1 on the split result so those sub-collectives
            # actually get relocated to their own earliest legal position
            # and their windows get filled, same as any other collective.
            split_module = _hlo.HloModule.from_serialized_hlo_module_proto(phase2_bytes)
            split_schedule = split_module.schedule()
            if split_schedule is not None:
                split_changed, split_split_candidates = _run_phase1_to_fixed_point(
                    split_module, split_schedule, module_name
                )
                if split_split_candidates:
                    _logger.info(
                        "collective_overlap_pass [%s]: %d further split "
                        "candidate(s) after re-optimizing the split result; "
                        "not cascading into another split round.",
                        module_name, len(split_split_candidates),
                    )
                if split_changed:
                    phase2_bytes = split_module.as_serialized_hlo_module_proto()
                _log_final_exposed_summary(
                    split_module, split_schedule, module_name, "post-phase-2 split_module"
                )
            return _dump_final_module(module_name, phase2_bytes)

    if changed:
        _log_final_exposed_summary(module, schedule, module_name, "phase-1-only module")
        return _dump_final_module(module_name, phase1_bytes)
    return None


def _collective_overlap_pass(serialized_hlo: bytes) -> Optional[bytes]:
    """Share rank 0's fully scheduled module with every rank.

    Every rank must compile a byte-identical executable. Rather than trying
    to keep the *inputs* to the (deterministic-in-theory) reorder/split
    algorithm in sync across ranks -- which proved insufficient: syncing
    _profile_costs alone still left ranks with divergent schedules, likely
    because the algorithm's output is sensitive to more than just the cost
    data -- we sidestep the whole class of divergence by only ever running
    the computation on rank 0 and sharing its *output* bytes with everyone
    else. Every rank (including rank 0) ends up returning the exact same
    bytes, so there is no way for schedules to diverge.

    The sync uses JAX's own distributed coordination-service key-value store
    (see jax/_src/compiler.py:_share_fdo_profiles for the stock-XLA analog:
    AutoPGLE syncs FDO profile bytes the exact same way).

    Both the KV-share key and the barrier names (below) are keyed by a
    per-process invocation counter (_barrier_invocation_count), not by a
    hash of the module content. Content-hashing was tried first (first the
    raw serialized_hlo bytes, then module.to_string()) and both broke in
    practice: two ranks' modules were repeatedly observed to be genuinely,
    provably semantically identical -- same shapes, same everything that
    matters for correctness -- yet still hash differently, because of
    non-deterministic-but-harmless serialization details with no bearing on
    program behavior (a per-compile-unique-id-like field in the raw proto
    in one case; a JSON key insertion order inside an opaque cuDNN
    backend_config string field in another -- e.g. {"24":"0","17":"1"} vs
    {"17":"1","24":"0"}, same tuning knobs, different order). Any such
    mismatch is permanent: rank 0 only ever publishes under the hash *it*
    computed, so a rank whose hash differs waits on a key that will never
    arrive. A call-order-based key sidesteps the entire class of problem --
    it doesn't care what's inside the module, only that every rank reaches
    this invocation in the same relative order, which we've verified
    empirically holds (matching ENTER/EXIT and barrier counts across ranks
    in every run so far).

    Rank 0 publishing while every other rank reads by itself is NOT enough
    to guarantee correctness, though: rank 0 never reads a key back (it
    only ever calls key_value_set_bytes, which returns as soon as the write
    lands -- it doesn't wait for anyone to consume it), so nothing stops
    rank 0 racing ahead through many further compiles/eager executions
    while other ranks sit blocked in blocking_key_value_get_bytes waiting
    for a not-yet-reached invocation. If one of those later, rank-0-only
    compiles eagerly executes an op needing a *real* cross-rank NCCL
    collective (observed: the MoE EP "borrowed comm" bootstrap creating the
    world clique), rank 0 ends up waiting on ranks that can never join,
    because they're each frozen earlier, waiting on a rank 0 that isn't
    coming back -- a genuine circular deadlock, distinct from the schedule
    divergence the key-value share otherwise fixes.

    The entry/exit barriers below (client.wait_at_barrier -- also a raw
    coordination-service RPC, safe to call from within this compiler
    callback, unlike jax.experimental.multihost_utils' barrier/broadcast
    helpers, which compile and run an actual jax.jit'd collective and would
    be an unsafe reentrant compile from in here) close that gap: no rank
    (rank 0 included) can leave one invocation before every rank has
    arrived at it and every rank has left it, so rank 0 can never get more
    than one invocation ahead of the slowest rank.
    """
    global _barrier_invocation_count

    # First statement in the function, deliberately as cheap as possible
    # (just os.getpid()/socket.gethostname(), nothing that touches JAX or
    # the distributed client yet) so this log line reliably fires the
    # instant XLA calls into this pass -- confirming the callback was
    # actually entered for this compile, as opposed to a hang happening
    # entirely inside XLA's C++ compiler *before* it ever reaches Python.
    _t_enter = time.monotonic()
    _logger.info(
        "collective_overlap_pass: ENTER host=%s pid=%d input_bytes=%d",
        socket.gethostname(), os.getpid(), len(serialized_hlo),
    )

    # Early-out before touching the distributed client/barrier at all: the
    # (much more common) modules with no async-done op anywhere -- tiny
    # utility ops like jit_broadcast_in_dim, jit_convert_element_type, etc.
    # -- have nothing for this pass to reorder, so there's no reason to pay
    # any synchronization cost for them at all.
    from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel
    module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
    if not _module_has_interesting_async_ops(module):
        _logger.info(
            "collective_overlap_pass: SKIP host=%s pid=%d module=%s (no "
            "async-done ops of interest; +%.1f ms since ENTER).",
            socket.gethostname(), os.getpid(), module.name,
            (time.monotonic() - _t_enter) * 1000,
        )
        _logger.info(
            "collective_overlap_pass: EXIT host=%s pid=%d result=no-op "
            "(skipped) (total %.1f ms since ENTER).",
            socket.gethostname(), os.getpid(),
            (time.monotonic() - _t_enter) * 1000,
        )
        return None

    client = _get_distributed_client()
    _in_rank = _jax_process_id() if client is not None else 0

    # Content-identity for this invocation -- diagnostic only, NOT used as
    # the KV-share/barrier key (content-hashing proved unreliable for that
    # and was replaced with the call-order-based _barrier_invocation_count;
    # see this function's docstring). Useful here as a quick way to spot,
    # by eye or `grep PRE-PASS INPUT HASH`, whether two ranks' modules for
    # the same invocation actually match, and if not, diff the text dumps.
    import hashlib as _hashlib  # pylint: disable=import-outside-toplevel
    _module_text = module.to_string()
    _content_digest = _hashlib.sha256(_module_text.encode()).hexdigest()

    # Dump the raw INPUT module -- exactly what XLA handed us, before any
    # pass logic (reorder/split/sync) has touched it -- unconditionally, for
    # every rank (not just rank 0, unlike the BEGIN ORIGINAL MODULE dump
    # inside _compute_collective_overlap, which only ever runs on rank 0).
    # This is what actually lets us diff whether two ranks' local HLO for
    # "the same" logical invocation is truly identical or not, rather than
    # inferring it indirectly from KV-share key mismatches.
    if os.environ.get("COLLECTIVE_OVERLAP_DUMP_MODULE", "1") == "1":
        _in_host = socket.gethostname()
        _in_pid = os.getpid()
        _logger.info(
            "collective_overlap_pass: PRE-PASS INPUT HASH rank=%d host=%s "
            "pid=%d module=%s sha256=%s bytes=%d text_bytes=%d",
            _in_rank, _in_host, _in_pid, module.name, _content_digest,
            len(serialized_hlo), len(_module_text),
        )
        sys.stderr.write(
            f"[collective_overlap_pass] === BEGIN PRE-PASS INPUT MODULE "
            f"(rank={_in_rank} host={_in_host} pid={_in_pid} "
            f"module={module.name} sha256={_content_digest}) ===\n"
            f"{_module_text}\n"
            f"[collective_overlap_pass] === END PRE-PASS INPUT MODULE "
            f"(rank={_in_rank} module={module.name}) ===\n"
        )

    is_root = client is None or _in_rank == 0
    multi_process = client is not None and _jax_process_count() > 1
    _logger.info(
        "collective_overlap_pass: resolved is_root=%s multi_process=%s "
        "(+%.1f ms since ENTER).",
        is_root, multi_process, (time.monotonic() - _t_enter) * 1000,
    )

    # Entry barrier: every rank (including rank 0) must arrive here before
    # any rank proceeds -- see the docstring for why this is the piece that
    # actually bounds rank 0's ability to race ahead.
    _entry_name = _exit_name = None
    if multi_process:
        _barrier_invocation_count += 1
        _entry_name = f"collective_overlap_pass_entry_{_barrier_invocation_count}"
        _exit_name = f"collective_overlap_pass_exit_{_barrier_invocation_count}"
        _rank_for_log = _jax_process_id()
        try:
            _logger.info(
                "collective_overlap_pass: ENTRY BARRIER BEGIN rank=%d "
                "name=%s.", _rank_for_log, _entry_name,
            )
            client.wait_at_barrier(_entry_name, _SHARE_TIMEOUT_MS)
            _logger.info(
                "collective_overlap_pass: ENTRY BARRIER END rank=%d "
                "name=%s.", _rank_for_log, _entry_name,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: entry barrier %s failed (%s); "
                "proceeding without it -- this reintroduces the pacing "
                "risk the barrier exists to close.", _entry_name, exc,
            )

    result_bytes: Optional[bytes] = None
    if is_root:
        _t_compute = time.monotonic()
        _logger.info("collective_overlap_pass: _compute_collective_overlap BEGIN.")
        result_bytes = _compute_collective_overlap(serialized_hlo)
        _logger.info(
            "collective_overlap_pass: _compute_collective_overlap END (%s, "
            "%.1f ms).",
            "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
            (time.monotonic() - _t_compute) * 1000,
        )

    if multi_process:
        _key = f"collective_overlap_pass_kv_{_barrier_invocation_count}"
        _rank_for_log = _jax_process_id()
        _host_for_log = socket.gethostname()
        _pid_for_log = os.getpid()
        try:
            if is_root:
                _payload = (
                    _SHARE_NONE if result_bytes is None else _SHARE_SOME + result_bytes
                )
                _logger.info(
                    "collective_overlap_pass: KV SET BEGIN rank=%d host=%s "
                    "pid=%d key=%s (%s).",
                    _rank_for_log, _host_for_log, _pid_for_log, _key,
                    "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
                )
                client.key_value_set_bytes(_key, _payload)
                _logger.info(
                    "collective_overlap_pass: KV SET END rank=%d host=%s "
                    "pid=%d key=%s.", _rank_for_log, _host_for_log, _pid_for_log, _key,
                )
            else:
                _logger.info(
                    "collective_overlap_pass: KV GET BEGIN rank=%d host=%s "
                    "pid=%d key=%s (waiting up to %d ms for process 0 to "
                    "share).", _rank_for_log, _host_for_log, _pid_for_log,
                    _key, _SHARE_TIMEOUT_MS,
                )
                _payload = client.blocking_key_value_get_bytes(_key, _SHARE_TIMEOUT_MS)
                result_bytes = (
                    None if _payload == _SHARE_NONE else _payload[len(_SHARE_SOME):]
                )
                _logger.info(
                    "collective_overlap_pass: KV GET END rank=%d host=%s "
                    "pid=%d key=%s (%s).",
                    _rank_for_log, _host_for_log, _pid_for_log, _key,
                    "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
                )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: sharing final scheduled module via "
                "the JAX distributed client failed (%s); falling back to "
                "this rank's own locally-computed result. This risks ranks "
                "ending up with divergent schedules.", exc,
            )
            if not is_root:
                result_bytes = _compute_collective_overlap(serialized_hlo)

        # Exit barrier: mirrors the entry barrier -- no rank leaves this
        # invocation until every rank has both arrived at and finished it,
        # so rank 0 can't move on to the next invocation (or to backend
        # compiling/eagerly executing this one) while a slower rank is
        # still waiting on the KV share above.
        try:
            _logger.info(
                "collective_overlap_pass: EXIT BARRIER BEGIN rank=%d "
                "name=%s.", _rank_for_log, _exit_name,
            )
            client.wait_at_barrier(_exit_name, _SHARE_TIMEOUT_MS)
            _logger.info(
                "collective_overlap_pass: EXIT BARRIER END rank=%d "
                "name=%s.", _rank_for_log, _exit_name,
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: exit barrier %s failed (%s); "
                "proceeding without it -- this reintroduces the pacing "
                "risk the barrier exists to close.", _exit_name, exc,
            )

    # Dump what THIS rank ends up returning, post-share, unconditionally
    # (unlike _dump_final_module inside _compute_collective_overlap, which
    # only ever runs on rank 0). Prefixing every line with rank/host/pid and
    # a content hash lets us grep across all ranks' log files and confirm
    # the share actually produced byte-identical modules everywhere --
    # e.g. `grep "POST-SHARE HASH" output-*.txt | sort | uniq -c` should
    # show exactly one distinct hash per module name if the share worked.
    if result_bytes is not None and os.environ.get(
        "COLLECTIVE_OVERLAP_DUMP_MODULE", "1"
    ) == "1":
        import hashlib as _hashlib  # pylint: disable=import-outside-toplevel
        _rank = _jax_process_id()
        _host = socket.gethostname()
        _pid = os.getpid()
        _digest = _hashlib.sha256(result_bytes).hexdigest()
        from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel
        _final_module = _hlo.HloModule.from_serialized_hlo_module_proto(result_bytes)
        _module_name = _final_module.name
        _logger.info(
            "collective_overlap_pass: POST-SHARE HASH rank=%d host=%s "
            "pid=%d module=%s sha256=%s bytes=%d",
            _rank, _host, _pid, _module_name, _digest, len(result_bytes),
        )
        sys.stderr.write(
            f"[collective_overlap_pass] === BEGIN POST-SHARE MODULE "
            f"(rank={_rank} host={_host} pid={_pid} module={_module_name} "
            f"sha256={_digest}) ===\n"
            f"{_final_module.to_string()}\n"
            f"[collective_overlap_pass] === END POST-SHARE MODULE "
            f"(rank={_rank} module={_module_name}) ===\n"
        )

    _logger.info(
        "collective_overlap_pass: EXIT host=%s pid=%d result=%s "
        "(total %.1f ms since ENTER).",
        socket.gethostname(), os.getpid(),
        "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
        (time.monotonic() - _t_enter) * 1000,
    )
    return result_bytes


def _dump_final_module(module_name: str, result_bytes: bytes) -> bytes:
    """Log the module as it will actually be handed back to XLA, if enabled."""
    if os.environ.get("COLLECTIVE_OVERLAP_DUMP_MODULE", "1") == "1":
        from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel
        _final_module = _hlo.HloModule.from_serialized_hlo_module_proto(result_bytes)
        sys.stderr.write(
            f"[collective_overlap_pass] === BEGIN FINAL MODULE: {module_name} ===\n"
            f"{_final_module.to_string()}\n"
            f"[collective_overlap_pass] === END FINAL MODULE: {module_name} ===\n"
        )
    return result_bytes


# ---------------------------------------------------------------------------
# PGLE profile interception
# ---------------------------------------------------------------------------
_patched = False


def _patch_pgle_profiler() -> None:
    global _patched
    if _patched:
        return
    _patched = True
    import jax._src.profiler as _jax_profiler  # pylint: disable=import-outside-toplevel
    _original_consume = _jax_profiler.PGLEProfiler.consume_fdo_profile

    def _consume_and_capture(self):
        result = _original_consume(self)
        if result:
            _update_profile(result)
        return result

    _jax_profiler.PGLEProfiler.consume_fdo_profile = _consume_and_capture
    _logger.debug("collective_overlap_pass: patched PGLEProfiler.consume_fdo_profile")

    # Also intercept the raw per-retry XSpace bytes, one level upstream of
    # consume_fdo_profile: PGLEProfiler.trace() calls
    # _profiler.get_fdo_profile(xspace) on each retry's raw XSpace and
    # discards xspace once it returns. Wrapping get_fdo_profile lets us
    # compute correct te_ep_* costs from that raw xspace before it's gone,
    # live within this run's own PGLE retries -- no reference trace needed.
    # _te_ep_overrides_loaded is set here too so
    # _apply_te_ep_cost_overrides (the reference-trace fallback) skips its
    # own load once live data is already in hand.
    global _te_ep_overrides_loaded
    _original_get_fdo_profile = _jax_profiler._profiler.get_fdo_profile

    def _get_fdo_profile_and_capture_te_ep(xspace):
        global _te_ep_overrides_loaded
        try:
            te_ep_costs = _load_te_ep_costs_from_xspace_bytes(bytes(xspace))
        except Exception as exc:  # pylint: disable=broad-except
            te_ep_costs = {}
            _logger.warning(
                "collective_overlap_pass: failed to extract te_ep_* costs "
                "from live PGLE profiling data: %s", exc,
            )
        if te_ep_costs:
            _te_ep_overrides.update(te_ep_costs)
            _profile_costs.update(te_ep_costs)
            _te_ep_overrides_loaded = True
            _logger.info(
                "collective_overlap_pass: captured %d live te_ep_* cost "
                "correction(s) from this run's own PGLE profiling data.",
                len(te_ep_costs),
            )
        return _original_get_fdo_profile(xspace)

    _jax_profiler._profiler.get_fdo_profile = _get_fdo_profile_and_capture_te_ep
    _logger.debug("collective_overlap_pass: patched _profiler.get_fdo_profile")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register() -> None:
    """Register the collective-overlap POST_SCHEDULER pass and PGLE hook."""
    # A hang inside XLA's own C++ compiler leaves no log lines and is
    # invisible to gdb/py-spy from outside the container (namespace entry
    # requires privileges we don't have on this cluster). faulthandler
    # sidesteps that: it writes directly to this process's own stderr on
    # receipt of a signal, so diagnosing a hang is just `kill -USR1 <pid>`
    # (e.g. `srun --overlap --jobid=<job> -w <host> kill -USR1 <pid>`)
    # using the rank/host/pid logged by this module's own log lines.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    _logger.info(
        "collective_overlap_pass: registered SIGUSR1 handler (faulthandler, "
        "all_threads=True) -- send SIGUSR1 to this process to dump every "
        "thread's Python traceback to stderr without needing gdb/"
        "container namespace access."
    )
    if os.environ.get("COLLECTIVE_OVERLAP_DISABLE", "0") == "1":
        _logger.info("collective_overlap_pass: disabled via COLLECTIVE_OVERLAP_DISABLE=1")
        return
    import jax.extend.xla as jex_xla  # pylint: disable=import-outside-toplevel
    _patch_pgle_profiler()
    jex_xla.register_hlo_module_transformation(
        _collective_overlap_pass,
        name="profile_guided_collective_overlap",
        stage=jex_xla.PipelineStage.POST_SCHEDULER,
    )
    _logger.info(
        "collective_overlap_pass: registered POST_SCHEDULER pass "
        "'profile_guided_collective_overlap'."
    )


# ---------------------------------------------------------------------------
# Subprocess entry point for Phase 2 (avoids descriptor pool conflicts)
# Usage: python collective_overlap_pass.py --split <hlo_file> <candidates_json>
# Writes modified HLO bytes to stdout; exits 0 on success, non-zero on error.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json as _json

    if len(sys.argv) >= 4 and sys.argv[1] == "--split":
        _hlo_file = sys.argv[2]
        _candidates_data = _json.loads(sys.argv[3])

        with open(_hlo_file, "rb") as _f:
            _serialized = _f.read()

        _candidates = [_SplitCandidate(**d) for d in _candidates_data]

        try:
            _result = _phase2_split_core(_serialized, _candidates)
        except Exception as _exc:
            import traceback as _tb
            sys.stderr.write(f"collective_overlap_pass split error: {_exc}\n")
            sys.stderr.write(_tb.format_exc())
            sys.exit(1)

        if _result:
            sys.stdout.buffer.write(_result)
        sys.exit(0)

    sys.stderr.write(f"Usage: {sys.argv[0]} --split <hlo_file> <candidates_json>\n")
    sys.exit(2)
