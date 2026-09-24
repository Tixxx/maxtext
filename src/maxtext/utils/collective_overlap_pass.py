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
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning("collective_overlap_pass: failed to parse FDO profile: %s", exc)


# ---------------------------------------------------------------------------
# Cross-rank sync (JAX's own distributed coordination service)
# ---------------------------------------------------------------------------
# Each rank's PGLE profile is measured independently from its own local NCCL/
# kernel timings, which differ slightly rank-to-rank due to hardware/timing
# noise. Since the pass's phase-1/phase-2 decisions (deficits, move targets,
# split groupings) are a deterministic function of _profile_costs, different
# _profile_costs across ranks can make different ranks schedule the *same*
# SPMD program differently -- which is illegal (all ranks must compile an
# identical executable).
#
# We used to fix this via an external cross-process broadcast library: first
# by broadcasting rank 0's _profile_costs (syncing the *input*, which turned
# out to be insufficient -- ranks still ended up with divergent schedules
# even when their profile-cost inputs matched), then by having only rank 0
# compute and broadcasting its *output* bytes (which sidesteps that, but
# requires the external library's process group to actually span all ranks --
# which in turn required extra Slurm launch flags that, on this cluster,
# interfered with NCCL's own cross-node bootstrap and caused a *different*,
# earlier hang).
#
# Tracing stock JAX/XLA's own AutoPGLE FDO-profile sync
# (jax/_src/compiler.py:_share_fdo_profiles) shows it solves this exact
# problem using infrastructure already built into JAX: JAX's own distributed
# coordination service (jax._src.distributed.global_state.client -- the same
# "Jax service"/"JAX distributed service" every one of these multi-process
# jobs is already connected to, since that's how the processes form a
# cluster in the first place) as a simple key-value store. A designated
# process publishes bytes under a content-derived key; every other process
# does a blocking read on that same key. No extra library, no barriers, no
# extra Slurm launch flags, no risk of colliding with NCCL's bootstrap. We do
# the same thing here, publishing the pass's *output* bytes (as established
# above, the output-broadcast strategy is the one that actually eliminates
# divergence) instead of the FDO profile.
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
    "get-tuple-element", "tuple", "copy",
})

_TRIVIAL_FUSED_OPCODES = frozenset({
    "parameter", "constant", "iota",
    "convert", "bitcast", "reshape", "transpose", "broadcast", "copy",
    "concatenate", "dynamic-slice",
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


_MAX_RELOCATE_CHAIN = 64

# Minimum profiled cost (us) for a non-trivial instruction to be considered
# a "heavy compute" worth relocating into an exposed collective's window --
# below this it's not worth the bookkeeping/relocation churn.
_HEAVY_COMPUTE_MIN_US = 5.0


def _earliest_legal_pos(
    ag_start,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
) -> tuple[int, list]:
    """Compute the earliest position ag_start can legally be relocated to.

    Walks ag_start's operands transitively through trivially-movable
    instructions (bitcast/reshape/elementwise ops, trivial fusions, etc.) —
    unlike a fixed relocation window, this walk has no distance bound, since
    a trivial/zero-cost op is free to move arbitrarily far back.  The walk
    stops at any non-trivial ("real compute") instruction, which pins a data-
    dependency floor: the move can be no earlier than right after it.  Every
    control-predecessor of a moved instruction (or of ag_start itself) that
    is not itself part of the moving set pins a control-dependency floor the
    same way.

    Returns (floor, movable) where floor is the smallest legal insertion
    position (in the current, pre-move schedule) and movable is the subset
    of the trivial chain that actually needs to relocate alongside ag_start
    (sorted in schedule order, excluding ag_start itself) -- i.e. those
    chain members currently positioned at or after floor.  Chain members
    already positioned before floor satisfy the ordering constraint as-is
    and are left untouched; forcing them to move too would needlessly drag
    ag_start's own final position later (floor + len(movable)), potentially
    past its original position, negating the point of the move.
    """
    candidates: set = set()
    seen: set = set()
    floor = 0
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
            floor = max(floor, positions[inst] + 1)

    moving_names = {inst.name for inst in candidates} | {ag_start.name}
    for inst in list(candidates) + [ag_start]:
        for name in _control_predecessor_names(inst):
            if name in moving_names:
                continue
            cp_pos = name_to_pos.get(name)
            if cp_pos is not None:
                floor = max(floor, cp_pos + 1)

    # floor only grows monotonically as the walk visits more of the operand
    # tree, so a candidate discovered early on may already sit before the
    # *final* floor -- only relocate the ones that don't.
    movable = [inst for inst in candidates if positions[inst] >= floor]
    return floor, sorted(movable, key=lambda i: positions[i])


def _fill_exposed_collectives_with_heavy_compute(
    seq: list,
    schedule,
    comp,
    start_of_done: dict,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    module_name: str,
) -> tuple[bool, list, dict, dict]:
    """Pull heavy compute instructions backward into earlier collectives'
    still-exposed [start, done) windows, to help hide their latency.

    Builds a bookkeeping list of every collective in `comp` that still has
    unhidden latency (deficit > 0) after the start-relocation pass above,
    ordered earliest-start-first. For each open window, scans forward from
    its `done` instruction for a heavy compute instruction (a real kernel --
    not a trivial bitcast/reshape/elementwise op -- with profiled cost at
    or above _HEAVY_COMPUTE_MIN_US) whose own real dependencies (found the
    same way _earliest_legal_pos finds them for a collective start) already
    sit at or before the window's `done` instruction, and relocates it into
    the window. Moving an instruction to an *earlier* position can never
    violate its own downstream consumers: in any valid schedule a consumer
    already sits after its producer, so it still sits after the new, even
    earlier, position too -- only upstream (operand/control-predecessor)
    dependencies need checking, which _earliest_legal_pos already does.
    Repeats per window until its deficit is closed or no more legal
    candidates remain, then moves on to the next window.
    """
    changed = False

    windows: list[dict] = []
    for ag_done, ag_start in start_of_done.items():
        if ag_start not in positions or ag_done not in positions:
            continue
        try:
            profile_key = ag_start.async_wrapped_root().name
        except Exception:
            profile_key = ag_start.name
        latency = _profile_costs.get(profile_key)
        if latency is None or latency <= 0:
            continue
        start_pos = positions[ag_start]
        done_pos = positions[ag_done]
        overlap = sum(
            _profile_costs.get(seq[i].name, 0.0) for i in range(start_pos + 1, done_pos)
        )
        deficit = latency - overlap
        if deficit > 0:
            windows.append({"start": ag_start, "done": ag_done, "deficit": deficit})

    if not windows:
        return False, seq, positions, name_to_pos

    windows.sort(key=lambda w: positions[w["start"]])
    excluded = set(start_of_done.keys()) | set(start_of_done.values())

    for window in windows:
        while window["deficit"] > 0:
            done_pos = positions[window["done"]]
            start_pos = positions[window["start"]]
            candidate = None
            cand_floor = None
            cand_chain: list = []
            for i in range(done_pos + 1, len(seq)):
                inst = seq[i]
                if inst in excluded:
                    continue
                if _is_trivially_movable_inst(inst, comp_by_name):
                    continue
                cost = _profile_costs.get(inst.name, 0.0)
                if cost < _HEAVY_COMPUTE_MIN_US:
                    continue
                floor, chain = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
                if floor > done_pos:
                    continue
                candidate = inst
                cand_floor = floor
                cand_chain = chain
                break

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

            cost = _profile_costs.get(candidate.name, 0.0)
            prev_deficit = window["deficit"]
            window["deficit"] = max(0.0, window["deficit"] - cost)
            _logger.info(
                "collective_overlap_pass [%s]: relocated heavy compute %s "
                "(cost %.1f us) into exposed window of %s (window deficit "
                "%.1f -> %.1f us).",
                module_name, candidate.name, cost, window["start"].name,
                prev_deficit, window["deficit"],
            )

    return changed, seq, positions, name_to_pos


# ---------------------------------------------------------------------------
# Phase 1: schedule reordering
# ---------------------------------------------------------------------------
_WHILE_CALLS_RE = re.compile(r"(?:condition|body)=%([A-Za-z0-9_.]+)")


def _innermost_first_computations(module, schedule) -> list:
    """Return non-fusion scheduled computations in innermost-first DFS order.

    While-body computations are visited before the computation that contains
    their while instruction, so that inner schedule changes are committed
    before outer schedules are processed.  Nested while loops are handled
    by recursing depth-first.  The entry computation is always last.
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

    # Build a map of computation → set of while-body children.
    # Simultaneously collect the set of all while-body callees so we can
    # identify the root (entry) computation as the one with no callers.
    #
    # inst.opcode is a jaxlib._hlo.HloOpcode enum (not a string) and
    # HloInstruction has no called_computations() accessor, so both the
    # opcode check and the callee lookup go through _opcode_str()/to_string()
    # parsing instead of direct attribute access.
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
        return all_comps

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
    return result


def _phase1_reorder(module, schedule, module_name: str) -> tuple[bool, list[_SplitCandidate]]:
    """Move async collective starts earlier where latency is under-hidden.

    Returns (changed, split_candidates) where split_candidates contains
    collectives that still had a deficit and were fully dep-blocked.
    """
    changed = False
    split_candidates: list[_SplitCandidate] = []
    # Module-wide name->computation map (includes fusion sub-computations,
    # unlike make_nonfusion_computations()), used to inspect a fusion's body
    # for triviality — see _is_trivial_fusion_body.
    comp_by_name = {c.name: c for c in module.computations()}

    for comp in _innermost_first_computations(module, schedule):
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

        for ag_done, ag_start in start_of_done.items():
            try:
                profile_key = ag_start.async_wrapped_root().name
            except Exception:
                profile_key = ag_start.name
            collective_latency = _profile_costs.get(profile_key)
            if collective_latency is None or collective_latency <= 0:
                _logger.debug(
                    "collective_overlap_pass [%s]: no profile entry for %s",
                    module_name, ag_start.name,
                )
                continue

            ag_start_pos = positions[ag_start]
            ag_done_pos = positions[ag_done]

            current_overlap = sum(
                _profile_costs.get(seq[i].name, 0.0)
                for i in range(ag_start_pos + 1, ag_done_pos)
            )

            if current_overlap >= collective_latency:
                _logger.debug(
                    "collective_overlap_pass [%s]: %s already hidden "
                    "(overlap=%.1f us >= latency=%.1f us).",
                    module_name, ag_start.name, current_overlap, collective_latency,
                )
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
            floor, to_move = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)

            if floor >= ag_start_pos:
                _logger.debug(
                    "collective_overlap_pass [%s]: %s cannot move "
                    "(no legal earlier position, deficit=%.1f us).",
                    module_name, ag_start.name, deficit,
                )
                if deficit >= _SPLIT_DEFICIT_THRESHOLD_US:
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
                _profile_costs.get(seq[i].name, 0.0)
                for i in range(new_ag_start_pos + 1, new_ag_done_pos)
            )
            _logger.info(
                "collective_overlap_pass: moving %s from pos %d to %d "
                "with %d relocated operand(s) (overlap %.1f -> %.1f us, "
                "latency=%.1f us).",
                ag_start.name, ag_start_pos, new_ag_start_pos,
                len(to_move), current_overlap, new_overlap, collective_latency,
            )

            if new_overlap < collective_latency:
                remaining_deficit = collective_latency - new_overlap
                if remaining_deficit >= _SPLIT_DEFICIT_THRESHOLD_US:
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
                    ))

        heavy_changed, seq, positions, name_to_pos = _fill_exposed_collectives_with_heavy_compute(
            seq, schedule, comp, start_of_done, positions, name_to_pos, comp_by_name, module_name,
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
                    _profile_costs.get(seq[i].name, 0.0)
                    for i in range(start_pos + 1, done_pos)
                )
                latency = overlap + cand.deficit_us
                remaining = latency - overlap
                if remaining >= _SPLIT_DEFICIT_THRESHOLD_US:
                    cand.deficit_us = remaining
                    still_needed.append(cand)
            split_candidates = still_needed

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

    # ID allocators
    #
    # Instruction IDs are packed as (computation_unique_id << 32) | local_id.
    # CalculateLocalId = id & 0xFFFFFFFF, used as key in each computation's
    # instruction_map.  Allocating simply from global_max+1 picks a value whose
    # lower 32 bits may equal an existing local_id in entry_comp → collision.
    #
    # Fix: for new instructions in entry_comp, use entry_comp's own parent bits
    # and a local_id above the current max in entry_comp.  For new async
    # computations, use their own comp id as the parent bits, starting at 0.
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
                # op_id itself only needs relocating when it is itself a
                # zero-cost op (bitcast/GTE/tuple) directly feeding the
                # collective-start — those must move together with the rest
                # of the chain so the sub-start's operands stay contiguous.
                # If op_id is instead the real (non-zero-cost) producer
                # (e.g. a GEMM feeding the collective with no zero-cost
                # wrapper), it must NOT be relocated: (1) moving a heavy
                # compute instruction can violate other consumers' ordering,
                # and (2) _proto_effective_pos(op_id) returns op_id's own
                # schedule position as eff_pos, and removing op_id from the
                # schedule invalidates that position in orig_to_compact,
                # which then silently falls back to the raw (pre-removal)
                # index — overshooting the true compact position badly
                # enough that the sub-done can end up inserted before its
                # own sub-start (RET_CHECK at hlo_schedule.cc:456).
                #
                # The dedup via _seen_gte_ids still applies: two different
                # operand indices in this (or another) group can reference
                # the exact same zero-cost instruction (e.g. a combined
                # all-gather whose operand tuple repeats a buffer), or one
                # operand's own op_id can turn out to be an ancestor
                # discovered while walking a later operand's chain. Either
                # way it must only be relocated once — inserting the same
                # instruction into the schedule twice trips XLA's schedule
                # verifier (hlo_schedule.cc:439).
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
            nr.dimensions.extend(inner_inst.dimensions)
            # Copy the replica-group spec verbatim. Modern XLA usually encodes
            # this via collective_device_list (or iota_collective_device_list)
            # rather than the legacy replica_groups field, which is then left
            # empty — copying only replica_groups silently drops the real
            # group, so the verifier falls back to inferring a full-device
            # subgroup (e.g. 32) instead of the instruction's true, possibly
            # smaller, subgroup (e.g. 8), tripping the shard_count ==
            # subgroup_size RET_CHECK in hlo_verifier.cc.
            nr.replica_groups.extend(inner_inst.replica_groups)
            # The modern replacement for replica_groups is the
            # "replica_group_list" oneof (collective_device_list /
            # iota_collective_device_list / mesh_axes_replica_group_list —
            # the latter is what Shardy-partitioned modules use). Whichever
            # variant is set, the participating-device grouping is identical
            # across all split sub-collectives (splitting only partitions
            # the operand/buffer list, not who talks to whom), so copy it
            # verbatim.
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
            # eff_pos should always land on a surviving instruction, but as
            # defense-in-depth (in case some other candidate/op ends up
            # removed at that exact position), snap to the nearest
            # surviving position at or before it rather than falling back
            # to the raw pre-removal index — using the raw index directly
            # can overshoot into (or past) compact positions reserved for
            # later insertions, e.g. the sub-done block, and cause the
            # RET_CHECK ordering violation seen in hlo_schedule.cc:456.
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

    # XLA's CreateFromProto processes computations in proto order and builds the
    # computation_map incrementally.  Callee computations must appear BEFORE their
    # callers.  The module entry computation must be last since it calls everything
    # else (including new async sub-computations added by phase 2).
    _module_entry_id = module_entry_comp.id
    _reordered = [c for c in proto.computations if c.id != _module_entry_id]
    _entry_protos = [c for c in proto.computations if c.id == _module_entry_id]
    _reordered.extend(_entry_protos)
    del proto.computations[:]
    for _c in _reordered:
        proto.computations.add().CopyFrom(_c)
    sys.stderr.write(
        f"[split_core] reordered computations: {len(_reordered)} total, "
        f"module entry last (id={_module_entry_id})\n"
    )

    return proto.SerializeToString()


def _phase2_split(serialized_hlo: bytes, candidates: list[_SplitCandidate]) -> Optional[bytes]:
    """Spawn a subprocess to run _phase2_split_core.

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

        if result.stdout:
            _logger.info(
                "collective_overlap_pass: split subprocess succeeded (%d bytes).",
                len(result.stdout),
            )
            return result.stdout

        _logger.info("collective_overlap_pass: split subprocess produced no output.")
        return None

    finally:
        try:
            os.unlink(_hlo_file)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Top-level POST_SCHEDULER pass entry point
# ---------------------------------------------------------------------------
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

    # ---- Phase 1 ----
    changed, split_candidates = _phase1_reorder(module, schedule, module_name)

    if changed:
        schedule.update()
        schedule.verify()
        module.set_schedule(schedule)
        phase1_bytes = module.as_serialized_hlo_module_proto()
    else:
        phase1_bytes = serialized_hlo

    # ---- Phase 2 ----
    if split_candidates:
        _logger.info(
            "collective_overlap_pass [%s]: %d split candidate(s) after phase 1.",
            module_name, len(split_candidates),
        )
        phase2_bytes = _phase2_split(phase1_bytes, split_candidates)
        if phase2_bytes is not None:
            return _dump_final_module(module_name, phase2_bytes)

    if changed:
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

    # Content-identity for this invocation -- diagnostic only (see
    # PRE-PASS INPUT HASH below), NOT used as the KV-share/barrier key
    # (see the docstring for why content-hashing, of either the raw
    # serialized_hlo bytes or this module.to_string() text, proved
    # unreliable for that and was replaced with the call-order-based
    # _barrier_invocation_count). Still useful here as a quick way to spot,
    # by eye or by `grep PRE-PASS INPUT HASH`, whether two ranks' modules
    # for the same invocation actually match or not, and if not, to diff
    # the accompanying text dumps to see exactly what differs.
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register() -> None:
    """Register the collective-overlap POST_SCHEDULER pass and PGLE hook."""
    # A hang inside XLA's own C++ compiler (as opposed to inside this
    # pass's Python code) leaves no further log lines and is invisible to
    # gdb/py-spy from outside the container (mount/pid namespace entry via
    # nsenter/enroot exec requires privileges we don't have on this
    # cluster). faulthandler sidesteps all of that: it writes directly to
    # this process's own stderr (captured in its output-*.txt like
    # everything else) on receipt of a signal, so diagnosing a hang is just
    # `kill -USR1 <pid>` from any session that can see the pid (e.g. `srun
    # --overlap --jobid=<job> -w <host> kill -USR1 <pid>`, no namespace
    # entry needed) using the rank/host/pid already logged by every
    # "... rank=%d host=%s pid=%d ..." line in this module.
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
