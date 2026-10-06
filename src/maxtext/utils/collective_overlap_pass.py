"""Profile-guided POST_SCHEDULER pass: move async collective starts earlier.

Phase 1 - schedule reordering:
  For every async collective in the scheduled HLO, use the PGLE FDO profile to
  check whether the ops between start and done hide its latency. If not, move
  the start earlier (bounded by data/control dependencies) and pull heavy
  compute into the exposed window.

Phase 2 - split batched collectives:
  A collective that still has a deficit because it bundles gradients from
  several layers is split into per-layer sub-collectives placed right after
  their producers, so each overlaps the next layer's compute.

Call ``register()`` once before the first ``jax.jit``-compiled function runs.
"""

from __future__ import annotations

import bisect
import faulthandler
import hashlib
import logging
import os
import re
import shutil
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

# Populated once PGLE delivers its FDO profile bytes.
_profile_costs: dict[str, float] = {}

# ---------------------------------------------------------------------------
# Proto compilation (PGLE profile + XLA HLO module), all in one directory to
# avoid descriptor pool conflicts.
# ---------------------------------------------------------------------------
_XLA_SRC = os.environ.get("DEFAULT_XLA_PATH") or "/opt/xla"
_TSL_SRC = f"{_XLA_SRC}/third_party/tsl"
_PROTO_OUT_DIR_BASE = "/tmp/_collective_overlap_pass_proto"

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


def _proto_out_dir() -> str:
    """Generated-code cache dir, keyed by protobuf runtime version and user so a
    directory produced by another container's protoc (or another user) is never
    reused."""
    from google.protobuf import __version__ as pb_version  # pylint: disable=import-outside-toplevel
    return f"{_PROTO_OUT_DIR_BASE}_{pb_version}_{os.getuid()}"


_GENCODE_CHECK_RE = re.compile(
    r"_runtime_version\.ValidateProtobufRuntimeVersion\(\s*_runtime_version\.Domain\.\w+,"
    r"\s*(\d+),\s*(\d+),\s*(\d+),[^)]*\)",
    re.S,
)


def _relax_gencode_version(root: str) -> int:
    """Drop the gencode-vs-runtime version check from generated modules whose
    gencode is newer than the Python protobuf runtime; returns how many files.

    The `protoc` on PATH can be newer than the installed runtime (e.g. gencode
    7.36.2 vs runtime 6.33.6), which makes every import raise VersionError. These
    are plain messages with no newer-protoc-only features, so they load fine on an
    older runtime once the check is gone.
    """
    from google.protobuf import __version__ as pb_version  # pylint: disable=import-outside-toplevel
    runtime = tuple(int(x) for x in re.findall(r"\d+", pb_version)[:3])
    patched = 0
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith("_pb2.py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path) as f:
                text = f.read()
            new = _GENCODE_CHECK_RE.sub(
                lambda m: "pass" if tuple(int(m.group(i)) for i in (1, 2, 3)) > runtime else m.group(0),
                text,
            )
            if new != text:
                with open(path, "w") as f:
                    f.write(new)
                patched += 1
    return patched


def _ensure_protos():
    """Compile all needed proto files once into a single output directory."""
    out = _proto_out_dir()
    marker = os.path.join(out, "xla", "service", "hlo_pb2.py")
    if not os.path.exists(marker):
        # Generate into a private directory and rename it into place, so ranks
        # racing on the same node never read a half-written tree.
        tmp = f"{out}.tmp.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        for rel in _PROTO_INIT_DIRS:
            d = os.path.join(tmp, rel)
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "__init__.py"), "a").close()
        try:
            subprocess.run(
                ["protoc",
                 f"--proto_path={_TSL_SRC}",
                 f"--proto_path={_XLA_SRC}",
                 "--proto_path=/usr/local/include",
                 f"--python_out={tmp}",
                 *_PROTO_SOURCES],
                check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            _logger.warning(
                "collective_overlap_pass: protoc failed: %s", exc.stderr.decode("utf-8", errors="replace")[-1000:],
            )
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        patched = _relax_gencode_version(tmp)
        if patched:
            _logger.warning(
                "collective_overlap_pass: protoc is newer than the Python protobuf runtime; "
                "removed the gencode version check from %d generated module(s).", patched,
            )
        try:
            os.rename(tmp, out)
        except OSError:  # another process finished first
            shutil.rmtree(tmp, ignore_errors=True)
    if out not in sys.path:
        sys.path.insert(0, out)


def _load_costs_from_fdo(fdo_bytes: bytes) -> dict[str, float]:
    _ensure_protos()
    from tsl.profiler.protobuf import profiled_instructions_pb2 as _pi  # type: ignore
    profiled = _pi.ProfiledInstructionsProto()
    profiled.ParseFromString(fdo_bytes)
    return {c.name: c.cost_us for c in profiled.costs}


def _update_profile(fdo_bytes: bytes) -> None:
    if not fdo_bytes:
        return
    try:
        costs = _load_costs_from_fdo(fdo_bytes)
        if not costs:
            return
        _profile_costs.update(costs)
        # PGLE's own te_ep entries are underestimates; keep the live corrections.
        _profile_costs.update(_te_ep_overrides)
        n_coll = sum(1 for k in costs if "all-gather" in k or "reduce-scatter" in k)
        _logger.info(
            "collective_overlap_pass: merged PGLE profile chunk: %d entries (%d collectives); pool now %d.",
            len(costs), n_coll, len(_profile_costs),
        )
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning("collective_overlap_pass: failed to parse FDO profile: %s", exc)


# ---------------------------------------------------------------------------
# Per-invocation cost corrections from the live PGLE XSpace
# ---------------------------------------------------------------------------
# PGLE's xplane->FDO conversion averages cost_us over every GPU kernel launch
# tagged with an HLO op's name. That is fine for single-kernel ops but wrong for
# ops that run as many kernels per execution (te_ep dispatch/combine/prepare:
# ~7 kernels, 2x-22x underestimate; grouped GEMMs ~5x; symmetric-memory
# collectives up to ~8x). _patch_pgle_profiler intercepts the raw XSpace and
# clusters events into invocations (a new one starts after a gap of
# _INVOCATION_GAP_US) to recover the per-invocation cost.
_TE_EP_PREFIX = "te_ep"
_INVOCATION_GAP_US = 100.0
# te_ep corrections replace the PGLE values in _profile_costs. The GEMM and
# collective corrections stay separate and are used only by the stream model
# (the rest of the pass keeps PGLE's numbers).
_te_ep_overrides: dict[str, float] = {}
_gemm_overrides: dict[str, float] = {}
_GEMM_COST_PREFIXES = ("te_gemm", "te_grouped_gemm")
_collective_overrides: dict[str, float] = {}
_COLLECTIVE_COST_PREFIXES = ("all-gather.", "reduce-scatter.", "all-reduce.")


def _average_invocation_costs(by_op: dict[str, list[tuple[float, float]]]) -> dict[str, float]:
    """{hlo_op: [(ts_us, dur_us), ...]} -> {hlo_op: mean per-invocation span}."""
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


def _load_costs_from_xspace_bytes(xspace_bytes: bytes, prefixes=None) -> dict[str, float]:
    """Per-invocation costs for HLO ops starting with `prefixes`, from a raw XSpace.

    Walks GPU device planes, resolves each event's `hlo_op` stat (event or event
    metadata; str_value or interned ref_value) and clusters by invocation.
    """
    _ensure_protos()
    from tsl.profiler.protobuf import xplane_pb2 as _xp  # type: ignore  # pylint: disable=import-outside-toplevel

    if prefixes is None:
        prefixes = (_TE_EP_PREFIX,)
    try:
        xspace = _xp.XSpace()
        xspace.ParseFromString(xspace_bytes)
    except Exception as exc:  # pylint: disable=broad-except
        _logger.warning("collective_overlap_pass: failed to parse live XSpace profile: %s", exc)
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
                if not hlo_op or not hlo_op.startswith(prefixes):
                    continue
                ts_us = (line.timestamp_ns * 1000 + event.offset_ps) / 1e6
                by_op.setdefault(hlo_op, []).append((ts_us, dur_us))

    return _average_invocation_costs(by_op)


# ---------------------------------------------------------------------------
# Cross-rank sync (JAX's coordination-service key-value store)
# ---------------------------------------------------------------------------
# All ranks must compile an identical executable, but each rank's PGLE profile
# differs slightly. Syncing the profile (the input) was not enough, and an
# external broadcast library needed Slurm flags that broke NCCL bootstrap. So
# rank 0 computes the pass and publishes its *output* bytes through
# jax._src.distributed.global_state.client (the same mechanism stock AutoPGLE
# uses for FDO profiles); everyone else does a blocking read.
_dist_client = None
_dist_checked = False

# Count of invocations that got past the early-out; names the barriers and the
# KV key (see _collective_overlap_pass for why not a content hash).
_barrier_invocation_count = 0


def _get_distributed_client():
    """Lazily resolve jax's distributed coordination-service client."""
    global _dist_client, _dist_checked
    if not _dist_checked:
        _dist_checked = True
        try:
            from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel
            _dist_client = _jax_distributed.global_state.client
            _process_id = _jax_distributed.global_state.process_id
            _process_count = _jax_distributed.global_state.num_processes
            _logger.info(
                "collective_overlap_pass: JAX distributed client: process_id=%d process_count=%d host=%s pid=%d.%s",
                _process_id, _process_count, socket.gethostname(), os.getpid(),
                "" if (_dist_client is not None and _process_count > 1) else (
                    " WARNING: no multi-process client; every rank will compute its own "
                    "(possibly divergent) schedule. Was jax.distributed.initialize() called?"
                ),
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: jax distributed client unavailable (%s); "
                "each rank will use its own schedule, which risks divergence.", exc,
            )
            _dist_client = None
    return _dist_client


def _jax_process_id() -> int:
    from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel
    return _jax_distributed.global_state.process_id


def _jax_process_count() -> int:
    from jax._src import distributed as _jax_distributed  # pylint: disable=import-outside-toplevel
    return _jax_distributed.global_state.num_processes


_SHARE_TIMEOUT_MS = int(
    os.environ.get("COLLECTIVE_OVERLAP_SHARE_TIMEOUT_MS", str(20 * 60 * 1000))
)

# Prefix bytes distinguishing "no transformation" from module bytes in the KV store.
_SHARE_NONE = b"\x00"
_SHARE_SOME = b"\x01"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Dedicated Start opcodes. Reduce-scatter (and te_ep calls) use the generic
# async-start wrapper instead, which is async by construction.
_ASYNC_START_OPCODES = frozenset({
    "all-gather-start",
    "all-reduce-start",
    "collective-permute-start",
})

# Minimum deficit (us) for phase-2 splitting. COLLECTIVE_OVERLAP_DISABLE_SPLIT=1
# disables it; that measured much worse (job 3193360: 149,096 us exposed vs
# 67,840 us with split; reproduced at full scale, job 3202509).
_SPLIT_DEFICIT_THRESHOLD_US = (
    float("inf") if os.environ.get("COLLECTIVE_OVERLAP_DISABLE_SPLIT") == "1"
    else 500.0
)

# COLLECTIVE_OVERLAP_WHILE_BODY_ONLY=1 skips the entry computation, which is by
# far the largest sequence and dominates compile time (deep chase chains, job
# 3204127's 20-minute timeout). Trades its overlap away for faster compiles.
_WHILE_BODY_ONLY = os.environ.get("COLLECTIVE_OVERLAP_WHILE_BODY_ONLY") == "1"

# COLLECTIVE_OVERLAP_HOIST_FSDP_STARTS=0 disables the final per-while-body step
# that moves FSDP starts (and their trivial operand chain) toward their
# earliest legal position.
_HOIST_FSDP_STARTS = os.environ.get("COLLECTIVE_OVERLAP_HOIST_FSDP_STARTS", "1") != "0"

# COLLECTIVE_OVERLAP_FSDP_STREAM_MODEL=0 disables the comm-stream model that
# places FSDP starts/dones in the hoist step (_best_fsdp_placement). While on,
# it also owns FSDP done placement in while bodies and
# _relocate_fsdp_done_before_te_ep is skipped there: that step left
# reduce-scatter.67/.68 with no compute to hide behind (job 3213143, 0% -> 100%
# exposed, ~198 ms/step).
_FSDP_STREAM_MODEL = os.environ.get("COLLECTIVE_OVERLAP_FSDP_STREAM_MODEL", "1") != "0"
# Simulated totals within this many us of the best are treated as equal when
# placing an FSDP start, so the earliest start (largest window) wins. Without
# it a ~30 us modeled gain moved reduce-scatter starts next to their dones
# (job 3213143).
_FSDP_STREAM_TOL_US = float(os.environ.get("COLLECTIVE_OVERLAP_FSDP_STREAM_TOL_US", "250"))
# SM gating: a launched heavy GEMM holds the SMs, so a collective kernel can't
# begin until it ends (job 3215273: simulated start times matched within
# ~0.1 ms). COLLECTIVE_OVERLAP_SM_GATE=0 disables; GEMMs shorter than
# COLLECTIVE_OVERLAP_SM_GATE_MIN_US are ignored.
_SM_GATE = os.environ.get("COLLECTIVE_OVERLAP_SM_GATE", "1") != "0"
_SM_GATE_MIN_US = float(os.environ.get("COLLECTIVE_OVERLAP_SM_GATE_MIN_US", "1000"))

# Names _phase2_split_core can split: plain batched collectives with a tuple
# operand list on the async-start. Matched on name, not opcode: all of these
# (and te_ep call-starts) share the generic "async-start" opcode.
_SPLITTABLE_NAME_PREFIXES = ("all-gather-start", "reduce-scatter-start", "all-reduce-start")

_SPLIT_MIN_GROUPS = 2
# Typical operands per layer epoch (34 operands / 5 FFN layers ~ 7).
_SPLIT_GROUP_SIZE = 7


@dataclass
class _SplitCandidate:
    start_name: str
    done_name: str
    deficit_us: float
    comp_name: str = ""
    # Schedule position of the real producer (through bitcasts/GTEs) of each
    # operand of the async-start.
    effective_producer_pos: list[int] = field(default_factory=list)
    # PGLE latency of the original collective; split sub-collectives have no
    # profile entry, so their latency is estimated as a byte share of this.
    total_latency_us: float = 0.0


def _opcode_str(inst) -> str:
    """Hyphenated opcode string (e.g. "get-tuple-element") for a phase-1 HloInstruction."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", inst.opcode.name[1:]).lower()


def _is_async_start(inst: object) -> bool:
    """True if `inst` is the Start half of an async collective.

    Deliberately permissive: the generic "async-start" opcode covers NCCL
    collectives (all-gather/reduce-scatter/all-reduce), te_ep call-starts and
    trivial passthrough wrappers alike, and there is no cheap reliable way to
    tell them apart by opcode/attributes. A trivial wrapper never shows a
    deficit worth acting on.
    """
    opcode = _opcode_str(inst)
    return opcode in _ASYNC_START_OPCODES or opcode == "async-start"


def _module_has_interesting_async_ops(module) -> bool:
    """True if any non-fusion computation has an async-start (cheap early-out)."""
    for comp in module.make_nonfusion_computations():
        for inst in comp.instructions():
            if _is_async_start(inst):
                return True
    return False


# XLA PrimitiveType -> bytes per element (xla_data.proto enum values)
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


_ZERO_COST_OPCODES = ("bitcast", "get-tuple-element", "tuple")


def _effective_producer_pos(inst, positions: dict, max_depth: int = 8) -> int:
    """Schedule position of inst's real producer, looking through zero-cost ops."""
    cur = inst
    for _ in range(max_depth):
        if _opcode_str(cur) not in _ZERO_COST_OPCODES:
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
    """_effective_producer_pos on proto instructions by ID (used by the split subprocess)."""
    cur_id = op_id
    for _ in range(max_depth):
        inst = id_to_inst.get(cur_id)
        if inst is None or inst.opcode not in _ZERO_COST_OPCODES:
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
# HLO text helpers (the phase-1 binding exposes little beyond name/opcode/
# operands/users/to_string, so the rest is parsed from to_string())
# ---------------------------------------------------------------------------
# "parameter"/"constant" are leaves with no real dependency, so always movable
# (job 3174209: their absence was the largest refusal reason blocking
# all-gather-start.8.g3's chase).
_TRIVIAL_SINGLE_OPCODES = frozenset({
    "bitcast", "convert", "transpose", "reshape", "broadcast",
    "get-tuple-element", "tuple", "copy", "slice",
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
_TO_APPLY_RE = re.compile(r"to_apply=%([A-Za-z0-9_.]+)")
_ROOT_RE = re.compile(r"\bROOT %([A-Za-z0-9_.\-]+) = ")
_CONTROL_PRED_RE = re.compile(r"control-predecessors=\{([^}]*)\}")


def _is_trivial_fusion_body(inst, comp_by_name: dict) -> bool:
    """True if every op in inst's fused computation is trivial."""
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
    """True if inst can be relocated without moving real compute."""
    opc = _opcode_str(inst)
    if opc in _TRIVIAL_SINGLE_OPCODES:
        return True
    if opc == "fusion":
        return _is_trivial_fusion_body(inst, comp_by_name)
    return False


def _control_predecessor_names(inst) -> list[str]:
    try:
        text = inst.to_string()
    except Exception:
        return []
    m = _CONTROL_PRED_RE.search(text)
    if not m:
        return []
    return [n.strip().lstrip("%") for n in m.group(1).split(",") if n.strip()]


def _computation_root(comp):
    """comp's ROOT instruction (the binding has no root_instruction() in this
    jaxlib; try it anyway, then parse "ROOT %name =", then fall back to the
    first user-less instruction, which can misfire on an unused parameter)."""
    try:
        return comp.root_instruction()
    except Exception:
        pass
    try:
        m = _ROOT_RE.search(comp.to_string())
        if m:
            for inst in comp.instructions():
                if inst.name == m.group(1):
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
    """Follow nested call/async wrappers down to the instruction PGLE profiles.

    Stops at the first non-call/async op (it may simply have no profile entry,
    as for a trivial passthrough).
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
    """Name under which ag_start's latency is keyed in _profile_costs.

    Standard collectives expose their wrapped root via async_wrapped_root().
    te_ep call-starts use to_apply=%name instead (async_wrapped_root() fails or
    returns an intermediate "call.N"), so fall back to parsing it. Either way
    the root is unwrapped further to the real leaf (te_ep_dispatch_ffi.N etc.).
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
    """True for a te_ep dispatch/combine call-start (the multi-ms MoE windows)."""
    if not _is_async_start(inst):
        return False
    return _resolve_profile_key(inst, comp_by_name).startswith(("te_ep_combine", "te_ep_dispatch"))


_TE_GEMM_TARGET_RE = re.compile(r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*)"')
_HEAVY_ANCHOR_TARGET_RE = re.compile(
    r'custom_call_target="(te_grouped_gemm[^"]*|te_gemm[^"]*|[^"]*cudnn[^"]*|[^"]*cublas[^"]*)"'
)


def _wraps_custom_call(inst, target_re, comp_by_name: dict) -> bool:
    """True if inst is a custom-call matching target_re, or a kind=kCustom
    fusion whose nested computation holds one (such a fusion never gets its own
    PGLE entry; the kernel is tagged with the inner custom-call's name)."""
    opc = _opcode_str(inst)
    try:
        text = inst.to_string()
    except Exception:
        return False
    if opc == "custom-call":
        return bool(target_re.search(text))
    if opc == "fusion" and "kind=kCustom" in text:
        m = _CALLS_RE.search(text)
        if not m:
            return False
        comp = comp_by_name.get(m.group(1))
        if comp is None:
            return False
        try:
            return any(
                _opcode_str(sub) == "custom-call" and target_re.search(sub.to_string())
                for sub in comp.instructions()
            )
        except Exception:
            return False
    return False


def _is_te_gemm_custom_call(inst, comp_by_name: dict) -> bool:
    """te_grouped_gemm/te_gemm kernel (the MoE GEMMs that dominate compute);
    prioritized when filling FSDP collective windows."""
    return _wraps_custom_call(inst, _TE_GEMM_TARGET_RE, comp_by_name)


def _is_heavy_anchor_custom_call(inst, comp_by_name: dict) -> bool:
    """te_gemm/cudnn/cublas kernel: where _chase_ag_start_blocker_toward_heavy_compute
    stops. Broader than _is_te_gemm_custom_call, kept separate so this chase
    path doesn't change the validated fill priority."""
    return _wraps_custom_call(inst, _HEAVY_ANCHOR_TARGET_RE, comp_by_name)


# FSDP-class collectives; all-reduce is handled like all-gather/reduce-scatter.
_FSDP_COLLECTIVE_PREFIXES = ("all-gather", "reduce-scatter", "all-reduce")

_MAX_RELOCATE_CHAIN = 64

# Minimum profiled cost (us) for an instruction to count as heavy compute.
_HEAVY_COMPUTE_MIN_US = 5.0


def _resolve_inst_cost(inst, comp_by_name: dict) -> float:
    """Profiled cost (us) of inst.

    A kind=kCustom fusion (e.g. dynamic-slice-fusion around te_gemm_v2_ffi plus
    a dynamic-update-slice) has no entry of its own, so sum the entries of
    every instruction in its nested computation.
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
    """Prefix sum of costs along `seq`, with `exclude`d instructions counted as 0."""
    prefix = [0.0] * (len(seq) + 1)
    for i, inst in enumerate(seq):
        c = 0.0 if inst in exclude else _resolve_inst_cost(inst, comp_by_name)
        prefix[i + 1] = prefix[i] + c
    return prefix


def _total_exposed_us(
    start_of_done: dict, positions: dict, seq: list, comp_by_name: dict,
) -> float:
    """Sum of positive deficits over every collective with a known latency, on
    the schedule as it stands: the metric that decides whether a move is a net win."""
    prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    total = 0.0
    for ag_done, ag_start in start_of_done.items():
        if ag_start not in positions or ag_done not in positions:
            continue
        latency = _profile_costs.get(_resolve_profile_key(ag_start, comp_by_name))
        if latency is None or latency <= 0:
            continue
        s, d = positions[ag_start], positions[ag_done]
        overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
        total += max(0.0, latency - overlap)
    return total


def _earliest_legal_pos(
    ag_start,
    positions: dict,
    name_to_pos: dict,
    comp_by_name: dict,
    cp_pins: Optional[list] = None,
) -> tuple[int, list, object]:
    """Earliest position ag_start can legally move to.

    Walks ag_start's operands transitively through trivially-movable
    instructions (no distance bound; trivial ops are free to move). The walk
    stops at any non-trivial instruction, which pins the floor to just after
    it. Control-predecessors outside the moving set pin it the same way, and
    are appended to `cp_pins` as (position, name) if a list is given.

    Returns (floor, movable, blocker): movable is the part of the trivial chain
    at or after the floor, in schedule order, excluding ag_start (chain members
    already before the floor stay put; moving them would drag ag_start's final
    position later). blocker is the non-trivial data producer pinning the
    floor, or None if the floor is 0 or pinned by a control-predecessor.
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
                blocker = None

    # The floor only grows as the walk proceeds, so early candidates may
    # already sit before the final one.
    movable = [inst for inst in candidates if positions[inst] >= floor]
    return floor, sorted(movable, key=lambda i: positions[i]), blocker


def _reinsert(seq: list, moving: list, at: int, last):
    """seq without `moving`/`last`, with `moving` then `last` inserted at `at`."""
    drop = set(moving) | {last}
    new_seq = [inst for inst in seq if inst not in drop]
    new_seq[at:at] = list(moving) + [last]
    return new_seq


def _commit_sequence(schedule, comp, new_seq: list):
    """Set comp's sequence; returns (seq, positions, name_to_pos)."""
    schedule.set_sequence(comp, new_seq)
    positions = {inst: i for i, inst in enumerate(new_seq)}
    name_to_pos = {inst.name: i for inst, i in positions.items()}
    return new_seq, positions, name_to_pos


def _relocate_fsdp_done_before_te_ep(
    ag_start, ag_done, comp, schedule, seq: list, positions: dict,
    name_to_pos: dict, comp_by_name: dict, collective_latency: float,
    module_name: str = "",
):
    """For an already-hidden FSDP collective, pull `done` forward to just before
    the first te_ep call-start at or after the point where its latency is
    closed, if any.

    `done` gates its consumers, so a later done than needed delays them for no
    benefit; an earlier one frees them to fill te_ep windows (the largest,
    hardest-to-fill ones). Always legal: done's only operand is start, and its
    consumers already sit after its old position. Control-predecessors are
    respected.

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

    new_seq, new_positions, new_name_to_pos = _commit_sequence(
        schedule, comp, _reinsert(seq, [], target, ag_done)
    )
    _logger.info(
        "collective_overlap_pass [%s]: moved %s done from pos %d to %d, before te_ep call-start %s "
        "(latency %.1f us closed by pos %d).",
        module_name, ag_done.name, done_pos, target, seq[target].name,
        collective_latency, latency_closure_pos,
    )
    return True, new_seq, new_positions, new_name_to_pos


# Bound on "blocker's own blocker" hops per chase. The chain is meant to be
# walked to the computation's inputs; each hop self-terminates when a blocker has
# no legal room, so this only guards against a pathological chain.
_MAX_PRODUCER_RELOCATE_HOPS = 64

# Bound on hops when chasing a fill candidate's blocker chain via
# _try_relocate_blocker_earlier (net-benefit-gated per hop, though cross-hop
# error can still accumulate). COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS=0 disables.
_MAX_HEAVY_COMPUTE_CHASE_HOPS = int(
    os.environ.get("COLLECTIVE_OVERLAP_HEAVY_CHASE_HOPS", "8")
)

# Outer fixed-point passes in _fill_exposed_collectives_with_heavy_compute: a
# fill or chase can expose a window that was hidden when the worklist was built.
_MAX_FILL_FIXED_POINT_ITERS = 3

# Give up on a blocked candidate once it has been the first blocked thing on
# this many consecutive scan restarts, so it can't starve reachable ones.
_STUCK_CANDIDATE_GIVE_UP_STREAK = 4


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
    """Position in [floor, blocker_pos] that minimizes TOTAL exposed time over
    every collective in the computation when `blocker` (and its trivial chain)
    moves there.

    Moving blocker to P changes ag_start's own overlap (its new floor is P +
    len(chain) + 1) and, for every other window, removes blocker's cost if it
    currently sits inside and wouldn't at P, or adds it in the reverse case.
    Chain members are trivial (~0 cost) and ignored. This is an approximation
    (pre-move positions decide window containment) but cheap, and it catches
    the dominant effect of stealing overlap from already-hidden windows.

    Candidates are blocker's *landing* positions, so "leave it" is unambiguously
    new_pos == blocker_pos (evaluating in insertion-start units made the no-op
    baseline depend on len(chain) and allowed oscillation between two blockers).
    Ties prefer the position closest to blocker_pos.

    Returns (best insertion-start position, best total exposed us). A result of
    blocker_pos - len(chain) means don't move.
    """
    prefix = _prefix_costs_excluding(seq, set(chain) | {blocker}, comp_by_name)
    block_cost = _resolve_inst_cost(blocker, comp_by_name)

    other_windows = []  # (start_pos, done_pos, latency, overlap excluding blocker)
    for od, os_ in start_of_done.items():
        if os_ is ag_start or os_ not in positions or od not in positions:
            continue
        latency2 = _profile_costs.get(_resolve_profile_key(os_, comp_by_name))
        if latency2 is None or latency2 <= 0:
            continue
        s2, d2 = positions[os_], positions[od]
        base_overlap = prefix[d2] - prefix[s2 + 1] if d2 > s2 + 1 else 0.0
        other_windows.append((s2, d2, latency2, base_overlap))

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
    """Relocate `blocker` (a non-trivial instruction pinning some collective's
    floor) and its trivial chain earlier, if it is heavy and one of two gates
    passes:

    1. Strict: the best position in [floor, current] strictly lowers TOTAL
       exposed time (see _find_best_blocker_position).
    2. Margin-safe fallback (only when `position_margins` is given, i.e. from
       the heavy-compute chase): move straight to the floor if blocker's cost
       fits within the slack of every collective covering its position. Path 1
       is myopic across a chain: moving a small op a few positions looks like a
       wash even when it is a necessary link to a candidate several hops away
       (job 3179392, input_reduce_fusion.60). Path 2 can never push a fine
       collective below its latency.

    Moving a producer earlier never breaks its consumers; the risk is stealing
    overlap another collective relied on, which both gates guard against.

    Returns the new (seq, positions, name_to_pos) if a move happened, else None.
    """
    if blocker not in positions:
        return None
    cost = _resolve_inst_cost(blocker, comp_by_name)
    if cost < _HEAVY_COMPUTE_MIN_US:
        return None
    floor, chain, _ = _earliest_legal_pos(blocker, positions, name_to_pos, comp_by_name)
    blocker_pos = positions[blocker]
    # The chain lands right before blocker, so its final position is
    # floor + len(chain). Comparing against the bare floor would report room to
    # move for a blocker already packed after its chain and loop forever.
    final_pos = floor + len(chain)
    if final_pos >= blocker_pos:
        return None

    best_pos, best_exposed = _find_best_blocker_position(
        blocker, chain, floor, blocker_pos, ag_start, ag_done_pos, ag_latency,
        start_of_done, positions, seq, comp_by_name,
    )
    current_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
    improves_total = (
        best_pos < blocker_pos - len(chain) and best_exposed < current_total - 1e-6
    )

    target = None
    if improves_total:
        target = best_pos
        _logger.debug(
            "collective_overlap_pass [%s]: TRC %s: pos %d -> %d (chain %d), total exposed %.1f -> %.1f us (path 1).",
            module_name, blocker.name, blocker_pos, best_pos + len(chain),
            len(chain), current_total, best_exposed,
        )
    elif position_margins is not None:
        margin = position_margins.get(blocker_pos, float("inf"))
        if cost <= margin:
            target = floor
            _logger.debug(
                "collective_overlap_pass [%s]: TRC %s: pos %d -> %d (chain %d), margin-safe "
                "(cost %.1f <= margin %.1f us) (path 2).",
                module_name, blocker.name, blocker_pos, floor + len(chain),
                len(chain), cost, margin,
            )

    if target is None:
        return None
    return _commit_sequence(schedule, comp, _reinsert(seq, chain, target, blocker))


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
    """FSDP-only: when ag_start has no earlier legal position because its direct
    non-trivial producer (`blocker`) sits right next to it, relocate that
    blocker earlier too, repeating until the blocker is a heavy anchor
    (_is_heavy_anchor_custom_call) or has no room.

    _earliest_legal_pos stops at the first non-trivial producer and never asks
    whether it can move, so "cannot move" can be misleading (job 3204236:
    all-gather-start.8.g0 was pinned by te_dbias_quantize_ffi.* while an
    unrelated GEMM sat comfortably earlier).

    Ungated (no net-exposure or margin check): an earlier gated version existed
    to protect other windows, but FSDP operand producers (quantize/dbias ops)
    are specific to that collective. If validation shows another collective
    regressing, the gate needs to come back.

    Returns (changed, seq, positions, name_to_pos, floor, to_move, blocker),
    with floor/to_move/blocker being ag_start's current _earliest_legal_pos.
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
        if b_floor + len(b_chain) >= blocker_pos:
            break
        seq, positions, name_to_pos = _commit_sequence(
            schedule, comp, _reinsert(seq, b_chain, b_floor, blocker)
        )
        changed = True
        hops += 1
        floor, to_move, blocker = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
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
    """If `blocker_done` is the done of ANOTHER still-exposed collective, try to
    move that collective's start earlier, so it sits between the caller's window
    and the heavy-compute candidate it is blocking.

    Narrower than the general recursive blocker chase (net-negative, job
    3150632): it fires only for a real dependency on another exposed
    collective's done. The move is capped at `window_done_pos` and bounded by
    the chased start's own legal floor.

    Returns (changed, seq, positions, name_to_pos); unchanged if no move.
    """
    if blocker_done not in exposed_dones:
        return False, seq, positions, name_to_pos
    chase_start = start_of_done.get(blocker_done)
    if chase_start is None or chase_start not in positions:
        return False, seq, positions, name_to_pos
    chase_start_pos = positions[chase_start]
    if chase_start_pos <= window_start_pos:
        return False, seq, positions, name_to_pos
    chase_floor, chase_chain, _ = _earliest_legal_pos(
        chase_start, positions, name_to_pos, comp_by_name
    )
    target = min(max(chase_floor, window_start_pos + 1), window_done_pos)
    if target + len(chase_chain) >= chase_start_pos:
        return False, seq, positions, name_to_pos

    new_seq, new_positions, new_name_to_pos = _commit_sequence(
        schedule, comp, _reinsert(seq, chase_chain, target, chase_start)
    )
    return True, new_seq, new_positions, new_name_to_pos


def _find_relocatable_ancestor(
    inst, positions: dict, name_to_pos: dict, comp_by_name: dict, max_depth: int,
    skip_cheap_ancestors: bool = False,
):
    """Walk `inst`'s blocker chain to the first ancestor with structural room to
    move (its floor + chain lands before its current position).

    Purely structural unless skip_cheap_ancestors is set; the result must still
    pass _try_relocate_blocker_earlier's gating before it is moved.

    skip_cheap_ancestors also requires cost >= _HEAVY_COMPUTE_MIN_US, so the
    walk passes over near-zero-cost connector ops (dynamic-slice/select
    fusions, a collective's own call-start/done) that the cost gate would
    refuse anyway (job 3202509). It is enabled for while bodies only: applied
    to `main` it found more relocations (79 -> 132) but made the final exposed
    total worse (72,880 -> 83,724 us, job 3203932 vs 3202899), because
    _find_best_blocker_position's per-hop approximation compounds over `main`'s
    long chains. In while bodies it measured a net win.

    Returns the ancestor (possibly `inst`), or None if the chain bottoms out,
    cycles or exhausts max_depth.
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
    """Relocate whatever blocks `candidate` from reaching a legal position in
    window [ag_start, ag_done_pos), up to `max_hops` times.

    The blocker can be any movable heavy op, e.g. dot_product_attention_fwd
    blocked by a GEMM blocked by another GEMM (the all-gather-start.8.g3 case).
    Each hop descends via _find_relocatable_ancestor to an op that can move, and
    hands it to _try_relocate_blocker_earlier, the sole committer. Reaching an
    indirect blocker typically takes two hops: the deep ancestor first, then the
    now-movable direct blocker on the next iteration.

    `position_margins` is forwarded to enable the margin-safe path, and
    recomputed after every committed hop so several hops can't cumulatively
    draw more than a window's slack.

    Stops on a refused hop, no relocatable ancestor, max_hops, or once the
    candidate's floor is resolved.

    Returns (changed, seq, positions, name_to_pos, floor, chain, blocker),
    with floor/chain/blocker being candidate's current _earliest_legal_pos.
    """
    changed = False
    floor, chain, blocker = _earliest_legal_pos(candidate, positions, name_to_pos, comp_by_name)
    hops = 0
    while floor > ag_done_pos and blocker is not None and hops < max_hops:
        target = _find_relocatable_ancestor(
            blocker, positions, name_to_pos, comp_by_name, max_hops - hops,
            skip_cheap_ancestors=is_while_body,
        )
        if target is None:
            break
        result = _try_relocate_blocker_earlier(
            target, comp, schedule, seq, positions, name_to_pos, comp_by_name,
            ag_start, ag_done_pos, ag_latency, start_of_done, module_name,
            position_margins,
        )
        hops += 1
        if result is None:
            break
        seq, positions, name_to_pos = result
        changed = True
        if position_margins is not None:
            position_margins = _compute_position_margins(
                start_of_done, positions, seq, comp_by_name
            )
        floor, chain, blocker = _earliest_legal_pos(
            candidate, positions, name_to_pos, comp_by_name
        )
    _logger.debug(
        "collective_overlap_pass [%s]: CHC %s: %s after %d/%d hop(s), floor=%d, done_pos=%d.",
        module_name, candidate.name, "resolved" if floor <= ag_done_pos else "gave up",
        hops, max_hops, floor, ag_done_pos,
    )
    return changed, seq, positions, name_to_pos, floor, chain, blocker


def _compute_position_margins(
    start_of_done: dict, positions: dict, seq: list, comp_by_name: dict,
    prefix: list | None = None, min_pos: int = 0,
) -> dict:
    """For every position >= min_pos, the minimum slack (overlap minus latency)
    among the collectives whose [start+1, done) window covers it: how much cost
    can be pulled out of that position without dropping any covering collective
    below its latency. Uncovered positions have no entry (callers treat that as
    infinite margin).

    Over-covered time is as safe to take from as idle time, as long as no more
    than the slack is taken (job 3179259: te_gemm_v2_ffi.81/.87/.78/.84 sat in
    all-gather-start.11's window with ~34 ms of slack and were wrongly
    off-limits).

    min_pos trims the work for a caller that only queries positions >= min_pos
    (the candidate scan, done_pos + 1). A caller that queries earlier
    positions, like the margin-safe fallback at a blocker's current position,
    must keep the default 0.
    """
    if prefix is None:
        prefix = _prefix_costs_excluding(seq, (), comp_by_name)
    margins: dict[int, float] = {}
    for ag_done, ag_start in start_of_done.items():
        if ag_start not in positions or ag_done not in positions:
            continue
        latency = _profile_costs.get(_resolve_profile_key(ag_start, comp_by_name))
        if latency is None or latency <= 0:
            continue
        s, d = positions[ag_start], positions[ag_done]
        if d <= s + 1 or d <= min_pos:
            continue
        slack = prefix[d] - prefix[s + 1] - latency
        for p in range(max(s + 1, min_pos), d):
            if p not in margins or slack < margins[p]:
                margins[p] = slack
    return margins


_GIVE_UP_MODES = ("none", "streak", "floor_aware")


class _StuckCandidateTracker:
    """Decides when a repeatedly blocked candidate is skipped so others get a
    turn. Strategies (compared per computation by
    _fill_exposed_collectives_best_of):

    - "none": never give up.
    - "streak": counts consecutive restarts blocked on the same candidate; a
      different name resets it. Cheap and safest in small while bodies (job
      3184713).
    - "floor_aware": per-candidate, counting a repeat only if its floor failed
      to improve. Better on the large entry computation (job 3187886), where
      multi-hop chains converge slowly.
    """

    def __init__(self, give_up_mode: str):
        self.give_up_mode = give_up_mode
        self._last_name: str | None = None
        self._last_streak = 0
        self._streak_by_name: dict[str, int] = {}
        self._last_floor_by_name: dict[str, int] = {}

    def note_and_maybe_give_up(self, name: str, floor: int) -> tuple[int, bool]:
        """Record that `name` was found blocked at `floor`; returns (streak, give_up)."""
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
    """Pull heavy compute backward into still-exposed collective windows.

    Outer fixed point (up to _MAX_FILL_FIXED_POINT_ITERS passes): each pass
    rebuilds the worklist from EVERY collective's fresh deficit, earliest start
    first, because filling or chasing one window can steal coverage from
    another that was already hidden (job 3181428: call-start.2/.4/.38 and
    all-gather-start.8.g2 became exposed purely as a side effect). A window
    whose deficit is unchanged since the last pass was already scanned to
    exhaustion and is skipped. Stops when a pass changes nothing.

    For each window, scan forward from its `done` for a heavy instruction
    (cost >= _HEAVY_COMPUTE_MIN_US, not trivial) whose real dependencies
    (_earliest_legal_pos) already sit at or before `done`, and move it into the
    window. This never breaks the candidate's consumers (they already sit after
    its producer). Candidates are restricted to those whose cost fits within
    the slack of the collectives covering their position
    (_compute_position_margins), so a direct placement can never push another
    collective below its latency. A candidate whose floor lies past `done` goes
    to the chase fallbacks instead (see _chase_blocked).

    The deficit is recomputed from the real prefix sums on every iteration. An
    incremental "deficit -= cost" double-credited instructions that a chase
    displaced and then re-placed.
    """
    changed = False
    pass_changed = False
    chase_hops = 0
    # Full-range margins for the chase fallbacks (a blocker's position can be
    # anywhere, unlike the scan's done_pos+1.. queries). Computed lazily once per
    # while-iteration and discarded after a successful chase mutates positions.
    chase_margins = None
    stuck_tracker = None
    given_up_candidates: set = set()
    prev_deficit_by_name: dict[str, float] = {}

    def _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones) -> bool:
        """`inst`'s floor lies past the window's done: record the stall and try
        the two chase fallbacks. Returns True if a chase moved something (the
        caller must restart its scan; positions have shifted).

        Fallback 1 relocates the start of another still-exposed collective whose
        done pins the candidate. Fallback 2 chases an arbitrary movable heavy
        blocker chain (_chase_heavy_compute_blocker_chain).
        """
        nonlocal seq, positions, name_to_pos, changed, pass_changed, chase_hops, chase_margins
        streak, give_up_now = stuck_tracker.note_and_maybe_give_up(inst.name, floor)
        if give_up_now:
            given_up_candidates.add(inst.name)
            _logger.debug(
                "collective_overlap_pass [%s]: window %s: giving up on %s after %d unresolved "
                "chase restarts (mode=%s, floor stuck at %d).",
                module_name, window["start"].name, inst.name, streak, give_up_mode, floor,
            )
        if blocker is not None and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
            moved, seq, positions, name_to_pos = _chase_exposed_collective_blocker_earlier(
                blocker, start_of_done, exposed_dones, start_pos, done_pos, seq,
                schedule, comp, positions, name_to_pos, comp_by_name,
            )
            if moved:
                chase_hops += 1
                changed = pass_changed = True
                _logger.info(
                    "collective_overlap_pass [%s]: chased exposed-collective blocker %s earlier "
                    "to unblock %s for window %s (deficit %.1f us).",
                    module_name, start_of_done[blocker].name, inst.name,
                    window["start"].name, window["deficit"],
                )
                return True
        if _MAX_HEAVY_COMPUTE_CHASE_HOPS > 0 and chase_hops < _MAX_PRODUCER_RELOCATE_HOPS:
            if chase_margins is None:
                chase_margins = _compute_position_margins(
                    start_of_done, positions, seq, comp_by_name,
                )
            moved, seq, positions, name_to_pos, _, _, _ = _chase_heavy_compute_blocker_chain(
                inst, comp, schedule, seq, positions, name_to_pos, comp_by_name,
                window["start"], done_pos, window["latency"], start_of_done,
                _MAX_HEAVY_COMPUTE_CHASE_HOPS, module_name, chase_margins,
                is_while_body=is_while_body,
            )
            if moved:
                chase_hops += 1
                changed = pass_changed = True
                return True
        return False

    for _fp_iter in range(_MAX_FILL_FIXED_POINT_ITERS):
        pass_changed = False

        windows: list[dict] = []
        cur_deficit_by_name: dict[str, float] = {}
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
                prev = prev_deficit_by_name.get(ag_start.name)
                windows.append({
                    "start": ag_start, "done": ag_done, "deficit": deficit,
                    "is_fsdp": profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES),
                    "latency": latency,
                    "skip": _fp_iter > 0 and prev is not None and abs(prev - deficit) < 1e-6,
                })

        prev_deficit_by_name = cur_deficit_by_name

        if not windows:
            break

        windows.sort(key=lambda w: positions[w["start"]])
        excluded = set(start_of_done.keys()) | set(start_of_done.values())
        # Only genuinely under-hidden collectives are chase targets, so
        # adequately overlapped ones aren't disturbed.
        exposed_dones = {w["done"] for w in windows}

        # Windows can overlap in time (concurrent async collectives, e.g.
        # region_19.46's call-start.56/.58); an instruction placed in a shared
        # zone counts toward both, so such placements are preferred. Best-effort
        # snapshot per pass.
        sibling_ranges: dict[str, list[tuple[int, int]]] = {}
        for w in windows:
            s, d = positions[w["start"]], positions[w["done"]]
            for other in windows:
                if other is w:
                    continue
                os_, od = positions[other["start"]], positions[other["done"]]
                if os_ < d and s < od:
                    sibling_ranges.setdefault(w["start"].name, []).append((os_, od))

        for window in windows:
            if window["skip"]:
                continue
            window_siblings = sibling_ranges.get(window["start"].name, [])
            chase_hops = 0
            # One window is drained fully before the next. Round-robin across
            # windows caused more total churn (job 3191562 nearly timed out).
            stuck_tracker = _StuckCandidateTracker(give_up_mode)
            given_up_candidates = set()
            while window["deficit"] > 0:
                done_pos = positions[window["done"]]
                start_pos = positions[window["start"]]
                iter_prefix = _prefix_costs_excluding(seq, (), comp_by_name)
                window["deficit"] = max(
                    0.0, window["latency"] - (iter_prefix[done_pos] - iter_prefix[start_pos + 1])
                )
                if window["deficit"] <= 0:
                    break
                # The scan only queries positions >= done_pos + 1.
                position_margins = _compute_position_margins(
                    start_of_done, positions, seq, comp_by_name, prefix=iter_prefix,
                    min_pos=done_pos + 1,
                )
                chase_margins = None
                candidate = None
                cand_floor = None
                cand_chain: list = []
                restart = False

                # FSDP windows prefer te_gemm candidates: of all legally
                # reachable ones take the earliest floor (it depends on the
                # dependency chain, not scan order), so the GEMM overlaps more
                # of the transfer. Falls through to the unrestricted scan if
                # none is reachable.
                if window["is_fsdp"]:
                    best_floor = None
                    for i in range(done_pos + 1, len(seq)):
                        inst = seq[i]
                        if inst in excluded or inst.name in given_up_candidates:
                            continue
                        if _is_trivially_movable_inst(inst, comp_by_name):
                            continue
                        if not _is_te_gemm_custom_call(inst, comp_by_name):
                            continue
                        cost = _resolve_inst_cost(inst, comp_by_name)
                        if cost < _HEAVY_COMPUTE_MIN_US:
                            continue
                        if cost > position_margins.get(i, float("inf")):
                            continue
                        floor, chain, blocker = _earliest_legal_pos(inst, positions, name_to_pos, comp_by_name)
                        if floor > done_pos:
                            if _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones):
                                restart = True
                                break
                            continue
                        if best_floor is None or floor < best_floor:
                            best_floor = floor
                            candidate = inst
                            cand_floor = floor
                            cand_chain = chain
                    if restart:
                        continue

                if candidate is None:
                    # Collect every legal candidate rather than taking the
                    # first: a 960 us GEMM spent on a 400 us gap strands the
                    # surplus another window needed. Best fit is picked below.
                    legal_candidates: list[tuple] = []
                    for i in range(done_pos + 1, len(seq)):
                        inst = seq[i]
                        if inst in excluded or inst.name in given_up_candidates:
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
                            if _chase_blocked(inst, floor, blocker, window, start_pos, done_pos, exposed_dones):
                                restart = True
                                break
                            continue
                        legal_candidates.append((inst, floor, chain, cost))
                    if restart:
                        continue

                    if legal_candidates:
                        def _lands_in_sibling(floor: int) -> bool:
                            return any(s <= floor < d for s, d in window_siblings)

                        sufficient = [c for c in legal_candidates if c[3] >= window["deficit"]]
                        if sufficient:
                            # Smallest candidate that closes the deficit,
                            # preferring one landing in a sibling window.
                            candidate, cand_floor, cand_chain, _ = min(
                                sufficient,
                                key=lambda c: (not _lands_in_sibling(c[1]), c[3]),
                            )
                        else:
                            candidate, cand_floor, cand_chain, _ = max(
                                legal_candidates,
                                key=lambda c: (_lands_in_sibling(c[1]), c[3]),
                            )

                if candidate is None:
                    break

                prev_deficit = window["deficit"]
                target = max(cand_floor, start_pos + 1)
                seq, positions, name_to_pos = _commit_sequence(
                    schedule, comp, _reinsert(seq, cand_chain, target, candidate)
                )
                changed = pass_changed = True
                _logger.info(
                    "collective_overlap_pass [%s]: relocated heavy compute %s (cost %.1f us) into "
                    "window of %s (pre-relocation deficit %.1f us).",
                    module_name, candidate.name, _resolve_inst_cost(candidate, comp_by_name),
                    window["start"].name, prev_deficit,
                )
        if not pass_changed:
            break
    return changed, seq, positions, name_to_pos


# ---------------------------------------------------------------------------
# Phase 1: schedule reordering
# ---------------------------------------------------------------------------
_WHILE_CALLS_RE = re.compile(r"(?:condition|body)=%([A-Za-z0-9_.]+)")
_WHILE_BODY_RE = re.compile(r"\bbody=%([A-Za-z0-9_.]+)")


def _innermost_first_computations(module, schedule) -> tuple[list, set]:
    """Return (computations, while_body_comps).

    computations are the scheduled non-fusion computations in innermost-first
    DFS order (a while body before the computation holding its while), the
    entry computation last. while_body_comps is every computation called as a
    while body, i.e. all but the true entry.
    """
    all_comps = [
        c for c in module.make_nonfusion_computations()
        if schedule.sequence(c) is not None
    ]
    # Restricted to this accessor call so lookups stay identity-consistent with
    # all_comps (HloComputation has no custom __eq__/__hash__).
    local_comp_by_name = {c.name: c for c in all_comps}

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

    roots = [c for c in all_comps if c not in all_callees]
    if not roots:
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
    for comp in all_comps:
        if comp not in visited:
            result.append(comp)
    return result, all_callees


def _log_profile_coverage_gaps(module, schedule, module_name: str) -> None:
    """Log fusion/custom-call instructions in while bodies with no PGLE cost
    entry; every cost lookup silently treats a missing one as free."""
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
    total = 0
    for comp in while_bodies:
        for inst in schedule.sequence(comp):
            opc = _opcode_str(inst)
            if opc not in ("fusion", "custom-call"):
                continue
            total += 1
            if inst.name not in _profile_costs:
                missing.append(f"{inst.name}({opc})")

    _logger.info(
        "collective_overlap_pass [%s]: profile coverage in %d while body(ies): %d/%d "
        "fusion/custom-call instructions have a PGLE cost (%d missing).",
        module_name, len(while_bodies), total - len(missing), total, len(missing),
    )
    if missing:
        _logger.info(
            "collective_overlap_pass [%s]: missing-profile instructions (first 150 of %d): %s",
            module_name, len(missing), ", ".join(missing[:150]),
        )


# Which _GIVE_UP_MODES strategy won per computation, so the ~3x-cost comparison
# runs once per _run_phase1_to_fixed_point call rather than on every re-entry
# (re-comparing every time timed out, job 3188145). Cleared at the start of
# each _run_phase1_to_fixed_point call.
_fill_strategy_cache: dict[str, str] = {}


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
    """Run the fill with each _GIVE_UP_MODES strategy from the same baseline and
    keep the one with the lowest _total_exposed_us.

    No strategy dominates: "streak" wins on small while bodies (job 3184713),
    "floor_aware" on the large entry computation (job 3187886). Re-entries for
    a computation replay the cached winner with a single trial.

    A per-window retry of stuck windows with the other modes was tried and
    reverted (job 3193061, identical deficits): all-gather-start.8.g0/.g3's
    candidates have a real dependency on all-gather-done.8.g2, a floor no
    give-up policy moves. That is a consequence of how the split grouped
    operands, not a fill gap.
    """
    cached_mode = _fill_strategy_cache.get(comp.name)
    if cached_mode is not None:
        pre_total = _total_exposed_us(start_of_done, positions, seq, comp_by_name)
        changed, seq, positions, name_to_pos = _fill_exposed_collectives_with_heavy_compute(
            list(seq), schedule, comp, start_of_done, dict(positions), dict(name_to_pos),
            comp_by_name, module_name, give_up_mode=cached_mode, is_while_body=is_while_body,
        )
        _logger.info(
            "collective_overlap_pass [%s]: comp %s: cached fill strategy '%s': total exposed "
            "%.1f -> %.1f us (changed=%s).",
            module_name, comp.name, cached_mode, pre_total,
            _total_exposed_us(start_of_done, positions, seq, comp_by_name), changed,
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
            "collective_overlap_pass [%s]: fill strategy '%s' for comp %s: total exposed %.1f us (changed=%s).",
            module_name, mode, comp.name, total, trial_changed,
        )
        if best is None or total < best[0] - 1e-6:
            best = (total, trial_changed, trial_seq, trial_positions, trial_name_to_pos, mode)

    total, changed, seq, positions, name_to_pos, mode = best
    schedule.set_sequence(comp, seq)
    _fill_strategy_cache[comp.name] = mode
    _logger.info(
        "collective_overlap_pass [%s]: comp %s: selected fill strategy '%s' (total exposed %.1f us).",
        module_name, comp.name, mode, total,
    )
    return changed, seq, positions, name_to_pos


def _may_hoist_control_pred(pred, positions, comp_by_name) -> bool:
    """Whether control-predecessor `pred` may move earlier.

    Anything may, except the done of an FSDP or te_ep collective. The recursive
    hoist reached such a done through a consumer right after it and pulled it
    next to its start (job 3213143's backward loop: reduce-scatter-done.1 from
    322 to 96, leaving no compute in the window).
    """
    opc = _opcode_str(pred)
    if not (opc == "async-done" or opc.endswith("-done")):
        return True
    ops = list(pred.operands())
    start = ops[0] if ops else None
    if start is None or start not in positions:
        return False
    try:
        key = _resolve_profile_key(start, comp_by_name)
    except Exception:
        key = ""
    return not key.startswith(_FSDP_COLLECTIVE_PREFIXES + (_TE_EP_PREFIX,))


def _simulate_comm_stream(order, cost: dict, lat: dict, done_start: dict, gate=None):
    """Simulate compute (in `order`) against one FIFO communication stream.

    Compute ops run back to back at their cost. Every async start is queued on
    the comm stream and begins at the later of its issue time and the end of the
    previous queued op; an async done stalls compute until its start has
    finished. Returns (total stall us, {done: stall us}).

    NCCL collectives and te_ep calls share one stream (job 3212637), so a
    collective issued behind a te_ep call waits for it and vice versa; the
    static window cost used elsewhere sees neither effect.

    `gate` is an optional set of SM-saturating GEMMs: a collective can't begin
    while one is running. The GEMM timeline depends on the stalls, which depend
    on the collective timeline, so it is iterated from the previous pass's GEMM
    intervals (a few passes converge).
    """
    intervals: list = []
    total = 0.0
    stalls: dict = {}
    for _ in range(3 if gate else 1):
        t = 0.0
        comm = 0.0
        end: dict = {}
        total = 0.0
        stalls = {}
        found: list = []
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
                for gs, ge in intervals:
                    if gs < begin < ge:
                        begin = ge
                comm = begin + lat[inst]
                end[inst] = comm
            else:
                c = cost.get(inst, 0.0)
                if gate and inst in gate:
                    found.append((t, t + c))
                t += c
        if not gate or found == intervals:
            break
        intervals = found
    return total, stalls


def _sim_inst_cost(inst, comp_by_name: dict) -> float:
    """_resolve_inst_cost, but with the per-invocation GEMM costs (_gemm_overrides)."""
    if not _gemm_overrides:
        return _resolve_inst_cost(inst, comp_by_name)
    direct = _gemm_overrides.get(inst.name)
    if direct:
        return direct
    if _opcode_str(inst) == "fusion":
        try:
            text = inst.to_string()
            if "kind=kCustom" in text:
                m = _CALLS_RE.search(text)
                comp = comp_by_name.get(m.group(1)) if m else None
                if comp is not None:
                    total = sum(
                        _gemm_overrides.get(sub.name) or _profile_costs.get(sub.name, 0.0)
                        for sub in comp.instructions()
                    )
                    if total:
                        return total
        except Exception:
            pass
    return _resolve_inst_cost(inst, comp_by_name)


def _best_fsdp_placement(
    ag_start, ag_done, floor, to_move, seq, positions, cost, lat, done_start, ctrl_succs,
    protect=None, gate=None, diag=None,
):
    """Choose where to put an FSDP start (and its done) to minimize simulated stalls.

    Candidates are insertion points for the start between its earliest legal
    position and the first consumer of its done. The done may stay put or move
    to just before its first consumer, so a start placed after its current done
    shifts the done along with it. Topological order is kept: the start after
    its operands and control-predecessors, the done after the start and before
    its consumers and control-successors. A start moving earlier brings its
    trivial operand chain (`to_move`); one moving later leaves it.

    Candidates within _FSDP_STREAM_TOL_US of the best simulated total count as
    equal; among them the earliest start wins, then the latest done (largest
    window; a later done never adds a stall in the sim).

    `protect` is an optional set of dones (the FSDP-class ones): a candidate is
    dropped if it raises their combined stall by more than the tolerance, so
    trading one FSDP collective's exposure for another's or a te_ep stall never
    counts as a gain.

    Returns ((total, done_moved, base_idx), moved_chain, start_idx, done_idx, base)
    or None; indices are into `base` (seq without the moved instructions).
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
    cands = []
    protected_now = 0.0
    if protect:
        _, stalls_now = _simulate_comm_stream(seq, cost, lat, done_start, gate)
        protected_now = sum(stalls_now.get(d, 0.0) for d in protect)
    chain = sorted(to_move, key=lambda x: positions[x])
    chain_pos = [positions[x] for x in chain]
    prepared: dict = {}

    def prepare(k):
        # Variant k moves chain[k:] with the start; chain[:k] already sit before
        # the insertion point and stay, so nothing moves later than it is now
        # (other users of a chain member may sit in between).
        if k not in prepared:
            rem = {ag_start, ag_done} | set(chain[k:])
            removed_before = [0] * (len(seq) + 1)
            count = 0
            for i, inst in enumerate(seq):
                removed_before[i] = count
                if inst in rem:
                    count += 1
            removed_before[len(seq)] = count
            prepared[k] = ([inst for inst in seq if inst not in rem], removed_before)
        return prepared[k]

    for early in (True, False):
        lo, hi = (floor, start_pos) if early else (start_pos + 1, p_max)
        hi = min(hi, p_max)
        for i in range(lo, hi + 1):
            # Interior points only matter next to heavy compute or an async op.
            if i < len(seq) and i not in (lo, hi, start_pos):
                inst = seq[i]
                if not (
                    cost.get(inst, 0.0) >= _HEAVY_COMPUTE_MIN_US
                    or inst in lat
                    or inst in done_start
                ):
                    continue
            k = bisect.bisect_left(chain_pos, i) if early else len(chain)
            base, removed_before = prepare(k)
            moved = chain[k:] if early else []
            q_latest = user_pos - removed_before[user_pos]
            q_cur = done_pos - removed_before[done_pos]
            pb = i - removed_before[i] if i < len(seq) else len(base)
            for q in {q_latest, q_cur}:
                if q < pb:
                    continue
                order = base[:pb] + moved + [ag_start] + base[pb:q] + [ag_done] + base[q:]
                total, stalls = _simulate_comm_stream(order, cost, lat, done_start, gate)
                if protect and sum(stalls.get(d, 0.0) for d in protect) > protected_now + _FSDP_STREAM_TOL_US:
                    if diag is not None:
                        diag.append((total, i, True))
                    continue
                if diag is not None:
                    diag.append((total, i, False))
                cands.append((total, q != q_cur, pb, k, q, moved, base))
    if not cands:
        return None
    limit = min(c[0] for c in cands) + _FSDP_STREAM_TOL_US
    total, _moved, pb, _k, q, moved, base = min(
        (c for c in cands if c[0] <= limit), key=lambda c: (c[2], -c[4])
    )
    return ((round(total, 1), _moved, pb), moved, pb, q, base)


# Recursion bound when hoisting a control-predecessor whose own latest operand
# must move first.
_MAX_PRED_HOIST_DEPTH = 8


def _hoist_fsdp_starts_to_floor(
    seq, schedule, comp, start_of_done, positions, name_to_pos, comp_by_name, module_name,
):
    """Move each FSDP start in a while body to its earliest legal position.

    Runs after the fill step. The per-collective loop in _phase1_reorder skips a
    collective once its static window looks hidden, and later fill/chase moves
    can land compute ahead of it with no dependency on it (job 3204509:
    all-gather-start.9 sat after te_grouped_quantize_ffi.177.double_buffer_clone
    and the dynamic_slice_fusion.25 GEMM). Its operand slices carry
    control-predecessors (loop_add_fusion.9/10, bitcasts of earlier
    all-gather-dones, DUS/async-done fusions) whose current positions set the
    floor even though they could sit much earlier. So while a control-predecessor
    pins the floor, hoist it to its own earliest legal position, then place the
    start and its trivial operand chain.

    With the stream model on, placement is chosen by _best_fsdp_placement
    (which may also move the done); otherwise the start goes to its floor and
    the done stays put, so the window only grows.
    """
    changed = False

    def _move_earlier(inst, depth) -> bool:
        """Move inst to just after its latest operand/control-predecessor,
        hoisting that one first (recursively, up to `depth`) if it sits right
        before inst."""
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
            if not _may_hoist_control_pred(inst, positions, comp_by_name):
                return False
            new_seq = [i for i in seq if i is not inst]
            if len(new_seq) != len(seq) - 1:
                _logger.warning(
                    "collective_overlap_pass [%s]: HOISTDIAG %s not found exactly once in sequence; skipping move.",
                    module_name, inst.name,
                )
                return False
            new_seq.insert(floor, inst)
            seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
            changed = True
            return True
        if binder is None or depth <= 0:
            return False
        if not _move_earlier(binder, depth - 1):
            return False
        return _move_earlier(inst, depth - 1)

    sim_cost: dict = {}
    sim_lat: dict = {}
    fsdp_dones: set = set()
    sim_gate: set = set()
    n_lat_overridden = 0
    ctrl_succs: dict = {}
    if _FSDP_STREAM_MODEL:
        for inst in seq:
            if _is_async_start(inst) or _opcode_str(inst).endswith("-done"):
                sim_cost[inst] = 0.0
            else:
                sim_cost[inst] = _sim_inst_cost(inst, comp_by_name)
                if (
                    _SM_GATE
                    and sim_cost[inst] >= _SM_GATE_MIN_US
                    and _is_te_gemm_custom_call(inst, comp_by_name)
                ):
                    sim_gate.add(inst)
            for name in _control_predecessor_names(inst):
                ctrl_succs.setdefault(name, []).append(inst)
        for d_inst, s_inst in start_of_done.items():
            try:
                key_s = _resolve_profile_key(s_inst, comp_by_name)
                sim_lat[s_inst] = (
                    _collective_overrides.get(key_s) or _profile_costs.get(key_s, 0.0) or 0.0
                )
                if key_s in _collective_overrides:
                    n_lat_overridden += 1
                if key_s.startswith(_FSDP_COLLECTIVE_PREFIXES):
                    fsdp_dones.add(d_inst)
            except Exception:
                sim_lat[s_inst] = 0.0
        _logger.info(
            "collective_overlap_pass [%s]: stream model: %s simulated total exposed %.1f us before FSDP placement "
            "(%d collective latencies from per-invocation clustering).",
            module_name, comp.name, _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done, sim_gate)[0],
            n_lat_overridden,
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
                break  # a data dependency pins the floor, not a control-predecessor
            if not _move_earlier(seq[pin_pos], _MAX_PRED_HOIST_DEPTH):
                _logger.debug(
                    "collective_overlap_pass [%s]: %s floor %d pinned by control-predecessor %s "
                    "(pos %d) which could not be hoisted.",
                    module_name, ag_start.name, floor, pin_name, pin_pos,
                )
                break
        floor, to_move, _blk = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
        start_pos = positions[ag_start]
        if _FSDP_STREAM_MODEL and sim_lat.get(ag_start, 0.0) > 0:
            before = _simulate_comm_stream(seq, sim_cost, sim_lat, start_of_done, sim_gate)[0]
            cand_diag: list = []
            best = _best_fsdp_placement(
                ag_start, ag_done, floor, to_move, seq, positions, sim_cost, sim_lat,
                start_of_done, ctrl_succs, protect=fsdp_dones, gate=sim_gate, diag=cand_diag,
            )
            if cand_diag:
                top = sorted(cand_diag)[:4]
                later = sorted((c for c in cand_diag if c[1] > start_pos))[:2]
                fmt = lambda cs: ", ".join(  # noqa: E731
                    "seq idx %d: %.1f us%s" % (c[1], c[0], " [dropped]" if c[2] else "") for c in cs
                ) or "none"
                _logger.debug(
                    "collective_overlap_pass [%s]: STREAMCANDS %s: %d candidates (%d dropped by FSDP-stall guard); "
                    "current %.1f us; lowest %s; lowest later than current %s.",
                    module_name, ag_start.name, len(cand_diag), sum(1 for c in cand_diag if c[2]), before,
                    fmt(top), fmt(later),
                )
            _logger.debug(
                "collective_overlap_pass [%s]: STREAMDIAG %s: start pos %d, done pos %d, floor %d, to_move %d, "
                "latency %.1f us, simulated total %.1f us; best %s.",
                module_name, ag_start.name, start_pos, positions[ag_done], floor, len(to_move),
                sim_lat.get(ag_start, 0.0), before,
                "none" if best is None else "total %.1f us at base idx %d, done idx %d, %d chain ops moved" % (
                    best[0][0], best[2], best[3], len(best[1])),
            )
            if best is not None:
                key, moved, pb, q, base = best
                if key[0] <= before + _FSDP_STREAM_TOL_US:
                    done_pos_old = positions[ag_done]
                    new_seq = (
                        base[:pb] + list(moved) + [ag_start]
                        + base[pb:q] + [ag_done] + base[q:]
                    )
                    if len(new_seq) == len(seq) and set(new_seq) == set(seq) and new_seq != seq:
                        seq, positions, name_to_pos = _commit_sequence(schedule, comp, new_seq)
                        changed = True
                        _logger.info(
                            "collective_overlap_pass [%s]: stream model placed %s: start pos %d -> %d, done pos %d -> %d "
                            "(simulated total exposed %.1f -> %.1f us).",
                            module_name, ag_start.name, orig_pos, positions[ag_start],
                            done_pos_old, positions[ag_done], before, key[0],
                        )
                continue
        if floor + len(to_move) >= start_pos:
            continue
        seq, positions, name_to_pos = _commit_sequence(
            schedule, comp, _reinsert(seq, to_move, floor, ag_start)
        )
        changed = True
        _logger.info(
            "collective_overlap_pass [%s]: hoisted %s from pos %d to %d with %d relocated operand(s).",
            module_name, ag_start.name, orig_pos, positions[ag_start], len(to_move),
        )
    return changed, seq, positions, name_to_pos


def _split_candidate_for(ag_start, ag_done, deficit, comp, positions, latency):
    """A _SplitCandidate for ag_start if its deficit warrants and it can be split, else None."""
    if deficit < _SPLIT_DEFICIT_THRESHOLD_US or not ag_start.name.startswith(_SPLITTABLE_NAME_PREFIXES):
        return None
    return _SplitCandidate(
        start_name=ag_start.name,
        done_name=ag_done.name,
        deficit_us=deficit,
        comp_name=comp.name,
        effective_producer_pos=[_effective_producer_pos(op, positions) for op in ag_start.operands()],
        total_latency_us=latency,
    )


def _phase1_reorder(module, schedule, module_name: str) -> tuple[bool, list[_SplitCandidate]]:
    """Move async collective starts earlier where latency is under-hidden.

    Returns (changed, split_candidates): collectives that still had a deficit
    and were blocked by dependencies.
    """
    changed = False
    split_candidates: list[_SplitCandidate] = []
    # Includes fusion sub-computations, unlike make_nonfusion_computations().
    comp_by_name = {c.name: c for c in module.computations()}

    ordered_comps, while_body_comps = _innermost_first_computations(module, schedule)
    for comp in ordered_comps:
        is_while_body = comp in while_body_comps
        if _WHILE_BODY_ONLY and not is_while_body:
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
                        "collective_overlap_pass [%s]: async-start %s has %d users (expected its done only); skipping.",
                        module_name, inst.name, len(users),
                    )

        if not start_of_done:
            continue

        positions = {inst: i for i, inst in enumerate(seq)}
        name_to_pos = {inst.name: i for inst, i in positions.items()}

        for ag_done, ag_start in start_of_done.items():
            profile_key = _resolve_profile_key(ag_start, comp_by_name)
            collective_latency = _profile_costs.get(profile_key)
            if collective_latency is None or collective_latency <= 0:
                _logger.debug(
                    "collective_overlap_pass [%s]: no profile entry for %s (resolved key: %s).",
                    module_name, ag_start.name, profile_key,
                )
                continue

            ag_start_pos = positions[ag_start]
            ag_done_pos = positions[ag_done]
            is_fsdp = profile_key.startswith(_FSDP_COLLECTIVE_PREFIXES)
            stream_owns_done = _FSDP_STREAM_MODEL and is_while_body

            current_overlap = sum(
                _resolve_inst_cost(seq[i], comp_by_name)
                for i in range(ag_start_pos + 1, ag_done_pos)
            )
            if current_overlap >= collective_latency:
                if is_fsdp and not stream_owns_done:
                    done_moved, seq, positions, name_to_pos = _relocate_fsdp_done_before_te_ep(
                        ag_start, ag_done, comp, schedule, seq, positions, name_to_pos,
                        comp_by_name, collective_latency, module_name,
                    )
                    changed = changed or done_moved
                continue

            deficit = collective_latency - current_overlap
            _logger.debug(
                "collective_overlap_pass [%s]: %s deficit=%.1f us (latency=%.1f us, overlap=%.1f us).",
                module_name, ag_start.name, deficit, collective_latency, current_overlap,
            )

            # Always move all the way to the floor: earlier only adds headroom
            # and puts the collective ahead of heavy compute it doesn't depend on.
            floor, to_move, blocker = _earliest_legal_pos(ag_start, positions, name_to_pos, comp_by_name)
            orig_ag_start_pos = ag_start_pos

            if is_fsdp:
                # The floor stops at the first non-trivial producer; try moving
                # that producer out of the way too.
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

            # The chain lands right before ag_start, so its post-move position
            # is floor + len(to_move); comparing the bare floor would treat an
            # already-packed start as movable.
            if floor + len(to_move) >= ag_start_pos:
                cand = _split_candidate_for(ag_start, ag_done, deficit, comp, positions, collective_latency)
                if cand is not None:
                    split_candidates.append(cand)
                continue

            seq, positions, name_to_pos = _commit_sequence(
                schedule, comp, _reinsert(seq, to_move, floor, ag_start)
            )
            changed = True

            new_overlap = sum(
                _resolve_inst_cost(seq[i], comp_by_name)
                for i in range(positions[ag_start] + 1, positions[ag_done])
            )
            _logger.info(
                "collective_overlap_pass: moving %s from pos %d to %d with %d relocated operand(s) "
                "(overlap %.1f -> %.1f us, latency=%.1f us).",
                ag_start.name, orig_ag_start_pos, positions[ag_start],
                len(to_move), current_overlap, new_overlap, collective_latency,
            )

            if new_overlap < collective_latency:
                cand = _split_candidate_for(
                    ag_start, ag_done, collective_latency - new_overlap, comp, positions,
                    collective_latency,
                )
                if cand is not None:
                    split_candidates.append(cand)
            elif is_fsdp and not stream_owns_done:
                done_moved, seq, positions, name_to_pos = _relocate_fsdp_done_before_te_ep(
                    ag_start, ag_done, comp, schedule, seq, positions, name_to_pos,
                    comp_by_name, collective_latency, module_name,
                )
                changed = changed or done_moved

        heavy_changed, seq, positions, name_to_pos = _fill_exposed_collectives_best_of(
            seq, schedule, comp, start_of_done, positions, name_to_pos, comp_by_name, module_name,
            is_while_body=is_while_body,
        )
        changed = changed or heavy_changed

        if is_while_body and _HOIST_FSDP_STARTS:
            hoist_changed, seq, positions, name_to_pos = _hoist_fsdp_starts_to_floor(
                seq, schedule, comp, start_of_done, positions, name_to_pos,
                comp_by_name, module_name,
            )
            changed = changed or hoist_changed

    return changed, split_candidates


# ---------------------------------------------------------------------------
# Phase 2: split batched collectives at proto level
# ---------------------------------------------------------------------------
def _group_operands_by_epoch(
    effective_positions: list[int],
) -> list[list[int]]:
    """Split operand indices into equal-rank groups ordered by effective producer position.

    Rank-based rather than gap-based, since per-layer GEMMs run back to back
    with no position gap. Returns [] if fewer than 2 groups would result.
    """
    n = len(effective_positions)
    n_groups = max(_SPLIT_MIN_GROUPS, (n + _SPLIT_GROUP_SIZE - 1) // _SPLIT_GROUP_SIZE)
    n_groups = min(n_groups, n // 2)  # at least 2 operands per group

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
    """Kahn topological sort of an instruction proto list.

    CreateFromProto needs every operand before its user; adding instructions and
    rerouting operands can break that. Falls back to the original order on cycles.
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

    Must run in a subprocess (hlo_pb2 conflicts with jaxlib's pre-registered
    protos in the descriptor pool). Its stderr is logged by the parent, which
    parses the GROUP_LATENCY lines.
    """
    if not candidates:
        return None

    _ensure_protos()
    from xla.service import hlo_pb2  # type: ignore  # pylint: disable=import-outside-toplevel
    from xla import xla_data_pb2  # type: ignore  # pylint: disable=import-outside-toplevel
    _TUPLE = xla_data_pb2.TUPLE  # = 13

    proto = hlo_pb2.HloModuleProto()
    proto.ParseFromString(serialized_hlo)

    id_to_comp = {c.id: c for c in proto.computations}
    name_to_comp = {c.name: c for c in proto.computations}

    # Module entry: the largest scheduled non-fusion computation.
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

    def _target_comp_for(cand):
        if cand.comp_name:
            c = name_to_comp.get(cand.comp_name)
            if c is None:
                sys.stderr.write(
                    f"[split_core] comp_name '{cand.comp_name}' not found, falling back to module entry\n"
                )
                return module_entry_comp
            return c
        return module_entry_comp

    # One invocation handles one computation's candidates (see _phase2_split).
    entry_comp = _target_comp_for(candidates[0])
    sys.stderr.write(f"[split_core] target comp: {entry_comp.name} ({len(entry_comp.instructions)} insts)\n")

    name_to_id = {inst.name: inst.id for inst in entry_comp.instructions}
    id_to_inst = {inst.id: inst for inst in entry_comp.instructions}

    sched_ids = list(proto.schedule.sequences[entry_comp.id].instruction_ids)
    id_to_sched_pos = {iid: pos for pos, iid in enumerate(sched_ids)}

    # Instruction IDs pack (computation_unique_id << 32) | local_id. Allocating
    # from global_max+1 can collide with an existing local_id in entry_comp, so
    # new instructions get entry_comp's parent bits and a fresh local id, and
    # new async computations use their own comp id as parent bits.
    _MASK32 = 0xFFFFFFFF
    _entry_parent_bits = entry_comp.id << 32
    _max_entry_local = max(
        (i.id & _MASK32 for i in entry_comp.instructions), default=-1
    )
    _next_entry_local = [_max_entry_local + 1]

    def _new_iid():
        v = _next_entry_local[0]
        _next_entry_local[0] += 1
        return _entry_parent_bits | (v & _MASK32)

    _next_comp_id = [max((c.id for c in proto.computations), default=0) + 1]

    def _new_cid():
        v = _next_comp_id[0]
        _next_comp_id[0] += 1
        return v

    def _new_async_iid(comp_parent_bits: int, local_ctr: list) -> int:
        v = local_ctr[0]
        local_ctr[0] += 1
        return comp_parent_bits | (v & _MASK32)

    # Fresh channel ids for the sub-collectives.
    _used_channels: set[int] = set()
    for _c in proto.computations:
        for _i in _c.instructions:
            if _i.channel_id:
                _used_channels.add(_i.channel_id)
    _next_channel_id = [max(_used_channels, default=0) + 1]

    def _new_channel():
        v = _next_channel_id[0]
        _next_channel_id[0] += 1
        return v

    any_split = False

    for cand in candidates:
        start_id = name_to_id.get(cand.start_name)
        if start_id is None:
            sys.stderr.write(f"[split_core] start {cand.start_name!r} not in proto\n")
            continue
        start_inst = id_to_inst.get(start_id)
        if start_inst is None:
            sys.stderr.write(f"[split_core] start_id {start_id} not in id_to_inst\n")
            continue

        done_id_search = name_to_id.get(cand.done_name)
        done_inst = id_to_inst.get(done_id_search) if done_id_search else None
        if done_inst is None:
            sys.stderr.write(f"[split_core] done {cand.done_name!r} not in proto\n")
            continue

        n_operands = len(start_inst.operand_ids)
        if n_operands < 2:
            sys.stderr.write(f"[split_core] {cand.start_name}: too few operands ({n_operands}), skip\n")
            continue

        if not start_inst.called_computation_ids:
            sys.stderr.write(f"[split_core] {cand.start_name}: no called_computation_ids\n")
            continue
        called_comp = id_to_comp.get(start_inst.called_computation_ids[0])
        if called_comp is None:
            sys.stderr.write(f"[split_core] {cand.start_name}: called comp not found\n")
            continue

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

        # Phase-1 positions may differ from the proto schedule's; they are used
        # only for grouping (relative order). Insertion points are recomputed
        # from the proto schedule below.
        groups = _group_operands_by_epoch(cand.effective_producer_pos)
        if not groups:
            sys.stderr.write(
                f"[split_core] {cand.start_name}: cannot form >=2 groups "
                f"(n={len(cand.effective_producer_pos)}, group_size={_SPLIT_GROUP_SIZE})\n"
            )
            continue

        # Per-operand bytes: each sub-collective's latency is its byte share of
        # the original's PGLE latency (they have no profile entry of their own,
        # so the post-split phase 1 would otherwise skip them). A byte-share
        # "skip the split if a group looks too small" heuristic was tried and
        # reverted: byte share doesn't predict downstream consumer count (job
        # 3191924/3193061: g2 had the smallest share but 7 consumers, g0 an
        # average share but 1).
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
            "collective_overlap_pass: splitting %s (deficit=%.1f us) into %d groups: %s",
            cand.start_name, cand.deficit_us, len(groups), epoch_summaries,
        )

        # --- Build one sub-collective per group ---
        # (group_indices, new_start_id, new_done_id, insert_after_pos, relocated_op_ids)
        new_pairs: list[tuple[list[int], int, int, int, list[int]]] = []
        _seen_gte_ids: set[int] = set()  # a zero-cost op must be relocated only once

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
                    if (inst is None or inst.opcode not in _ZERO_COST_OPCODES
                            or not inst.operand_ids):
                        break
                    next_id = inst.operand_ids[0]
                    next_inst = id_to_inst.get(next_id)
                    if next_inst is None or next_inst.opcode not in _ZERO_COST_OPCODES:
                        break
                    if next_id not in _seen_gte_ids:
                        intermediates.append(next_id)
                        _seen_gte_ids.add(next_id)
                    cur_id = next_id
                group_op_ids.extend(reversed(intermediates))
                # op_id moves only if it is a zero-cost op feeding the start, so
                # the sub-start's operands stay contiguous. A real producer (a
                # GEMM) must NOT move: it can violate other consumers' ordering,
                # and removing it invalidates its cached position in
                # orig_to_compact, overshooting enough to put the sub-done before
                # its sub-start (RET_CHECK at hlo_schedule.cc:456). The dedup
                # also avoids a duplicate insertion tripping hlo_schedule.cc:439.
                _op_inst = id_to_inst.get(op_id)
                if (_op_inst is not None and _op_inst.opcode in _ZERO_COST_OPCODES
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

            # Inner collective (same opcode / dims / replica groups as the original)
            new_rs_id = _new_async_iid(_nc_parent_bits, _nc_local_ctr)
            nr = nc.instructions.add()
            nr.id = new_rs_id
            nr.name = f"{inner_inst.name}.g{g_idx}"
            nr.opcode = inner_inst.opcode
            nr.operand_ids.extend(new_param_ids)

            _group_bytes = sum(_operand_bytes[i] for i in group)
            _group_latency_us = cand.total_latency_us * (_group_bytes / _total_operand_bytes)
            sys.stderr.write(f"[split_core] GROUP_LATENCY {nr.name} {_group_latency_us:.6f}\n")
            nr.dimensions.extend(inner_inst.dimensions)
            # Copy the device grouping verbatim, whichever form it takes: modern
            # XLA leaves the legacy replica_groups empty and uses the
            # replica_group_list oneof (collective_device_list /
            # iota_collective_device_list / mesh_axes_replica_group_list).
            # Dropping it makes the verifier infer a full-device subgroup and
            # trip the shard_count == subgroup_size RET_CHECK.
            nr.replica_groups.extend(inner_inst.replica_groups)
            _which_dl = inner_inst.WhichOneof("replica_group_list")
            if _which_dl is not None:
                getattr(nr, _which_dl).CopyFrom(getattr(inner_inst, _which_dl))
            nr.use_global_device_ids = inner_inst.use_global_device_ids
            # collective-permute defines participants by source_target_pairs.
            if inner_inst.opcode == "collective-permute":
                nr.source_target_pairs.extend(inner_inst.source_target_pairs)
            nr.called_computation_ids.extend(inner_inst.called_computation_ids)  # to_apply
            if inner_inst.channel_id:
                nr.channel_id = _new_channel()
            nr.metadata.CopyFrom(inner_inst.metadata)
            if inner_inst.backend_config:
                nr.backend_config = inner_inst.backend_config
            nr.shape.element_type = _TUPLE
            for orig_idx in group:
                s = nr.shape.tuple_shapes.add()
                s.CopyFrom(inner_inst.shape.tuple_shapes[orig_idx])
            nc.root_id = new_rs_id

            # Every non-fusion computation needs a schedule sequence.
            nc_seq = proto.schedule.sequences[new_cid]
            for _p in nc.instructions:
                nc_seq.instruction_ids.append(_p.id)

            # async-start in the target computation
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
            # Shape: (context_tuple, output_tuple), both of TUPLE type.
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

            # async-done in the target computation
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
            nd.shape.element_type = _TUPLE
            for orig_idx in group:
                s = nd.shape.tuple_shapes.add()
                s.CopyFrom(start_inst.shape.tuple_shapes[1].tuple_shapes[orig_idx])

            new_pairs.append((group, ns_id, nd_id, effective_insert_after, group_op_ids))

        # --- Reroute GTE users of the old done to the matching split done ---
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

        # Zero-cost operands being repositioned with their sub-start.
        _all_group_op_ids: set[int] = set()
        for (_, _, _, _, op_ids) in new_pairs:
            _all_group_op_ids.update(op_ids)

        _remove_from_sched = {start_inst.id, done_inst.id} | _all_group_op_ids
        new_sched = [iid for iid in sched_ids if iid not in _remove_from_sched]

        # Original positions -> compact positions (after all removals).
        orig_to_compact: dict[int, int] = {}
        _cidx = 0
        for _oi, _iid in enumerate(sched_ids):
            if _iid not in _remove_from_sched:
                orig_to_compact[_oi] = _cidx
                _cidx += 1

        def _compact_pos_at_or_before(pos: int) -> int:
            # Snap to the nearest surviving position at or before `pos`: the raw
            # pre-removal index can overshoot into positions reserved for the
            # sub-dones and trip the ordering RET_CHECK at hlo_schedule.cc:456.
            for _p in range(pos, -1, -1):
                if _p in orig_to_compact:
                    return orig_to_compact[_p]
            return 0

        # Insert each group's zero-cost operands + sub-start right after its producer.
        sorted_pairs = sorted(new_pairs, key=lambda x: x[3])
        offset = 0
        for group, ns_id, nd_id, eff_pos, op_ids in sorted_pairs:
            ins_pos = _compact_pos_at_or_before(eff_pos) + 1 + offset
            for _op_id in op_ids:
                new_sched.insert(ins_pos, _op_id)
                ins_pos += 1
                offset += 1
            new_sched.insert(ins_pos, ns_id)
            offset += 1

        # Sub-dones go just before the first surviving instruction after the old done.
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

        id_to_inst = {inst.id: inst for inst in entry_comp.instructions}
        name_to_id = {inst.name: inst.id for inst in entry_comp.instructions}

        any_split = True

    if not any_split:
        return None

    # CreateFromProto builds computation_map incrementally, so every callee must
    # precede its caller. entry_comp may be a while body rather than the module
    # entry and gains new callees (the sub-computations, appended at the end),
    # so a "module entry last" rule is not enough (job 3203819: "all-gather-
    # start.6.g0 instruction references invalid computation id(s)"). Post-order
    # DFS over the call graph handles both cases.
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

    return proto.SerializeToString()


_GROUP_LATENCY_RE = re.compile(r"^\[split_core\] GROUP_LATENCY (\S+) ([0-9.eE+-]+)")


def _phase2_split_one_comp(
    serialized_hlo: bytes, candidates: list[_SplitCandidate]
) -> Optional[tuple[bytes, dict[str, float]]]:
    """Run _phase2_split_core in a subprocess for one computation's candidates.

    _phase2_split_core resolves its target from candidates[0].comp_name and
    drops candidates from any other computation, so callers must pre-group by
    comp_name. Returns (module bytes, {sub-collective name: latency us}).
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

        _sub_stderr = result.stderr.decode("utf-8", errors="replace").strip()
        if _sub_stderr:
            _logger.info("collective_overlap_pass: split subprocess stderr:\n%s", _sub_stderr)

        group_latencies: dict[str, float] = {}
        for _line in _sub_stderr.splitlines():
            _m = _GROUP_LATENCY_RE.match(_line)
            if _m:
                group_latencies[_m.group(1)] = float(_m.group(2))

        if result.stdout:
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
    """Split every candidate across however many computations they span.

    Groups by comp_name, runs one subprocess per group and chains each group's
    output into the next. Returns None only if every group failed; otherwise
    the accumulated module and the merged sub-collective latency estimates
    (which the caller must feed into _profile_costs before re-running phase 1).
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
                "collective_overlap_pass: split produced no output for comp '%s' (%d candidate(s)); leaving them unsplit.",
                comp_name or "<module entry>", len(group),
            )

    return (current_bytes, all_group_latencies) if any_succeeded else None


# ---------------------------------------------------------------------------
# Top-level POST_SCHEDULER pass
# ---------------------------------------------------------------------------
# Cap on phase-1 re-runs against a freshly update()'d module. Normal runs
# converge in 1-2 iterations.
_MAX_PHASE1_FIXED_POINT_ITERS = 4


def _log_final_exposed_summary(module, schedule, module_name: str, label: str) -> None:
    """Log total and per-collective exposed time computed directly from the live
    schedule: ground truth, since intermediate totals go stale after a later
    schedule.update()/verify()."""
    comp_by_name = {c.name: c for c in module.computations()}
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

        prefix = _prefix_costs_excluding(seq, (), comp_by_name)
        comp_total = 0.0
        for ag_done, ag_start in start_of_done.items():
            if ag_start not in positions or ag_done not in positions:
                continue
            latency = _profile_costs.get(_resolve_profile_key(ag_start, comp_by_name))
            if latency is None or latency <= 0:
                continue
            s, d = positions[ag_start], positions[ag_done]
            overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
            deficit = max(0.0, latency - overlap)
            if deficit > 0:
                per_collective.append((comp.name, ag_start.name, deficit))
                comp_total += deficit
        if comp_total > 0:
            per_comp.append((comp.name, comp_total))
    per_comp.sort(key=lambda x: -x[1])
    per_collective.sort(key=lambda x: -x[2])
    _logger.info(
        "collective_overlap_pass [%s]: FINAL exposed summary [%s]: grand_total=%.1f us across %d computation(s): %s",
        module_name, label, sum(t for _, t in per_comp), len(per_comp),
        ", ".join(f"{name}={total:.1f}us" for name, total in per_comp),
    )
    _logger.info(
        "collective_overlap_pass [%s]: FINAL per-collective exposed [%s]: %s",
        module_name, label,
        ", ".join(f"{comp_name}/{name}={deficit:.1f}us" for comp_name, name, deficit in per_collective),
    )


def _run_phase1_to_fixed_point(module, schedule, module_name: str) -> tuple[bool, list[_SplitCandidate]]:
    """Run _phase1_reorder to a fixed point.

    schedule.update()/verify() (needed so XLA can canonicalize the schedule after
    raw seq mutations) can itself relocate an instruction whose dependency one
    of our earlier moves invalidated, without _phase1_reorder's bookkeeping
    noticing, so a window scored hidden can end up exposed (job 3161055:
    all-gather-start.9's window vanished after update() because
    te_gemm_v2_ffi.93 depends on all-gather-done.8). Re-running phase 1 on the
    post-update schedule re-fills it.

    Also used on a module straight out of phase 2: the split is pure proto
    surgery, so the new sub-collectives sit wherever the topological fixup put
    them, not at their earliest legal position.
    """
    global _fill_strategy_cache
    _fill_strategy_cache = {}
    changed = False
    split_candidates: list[_SplitCandidate] = []
    for _ in range(_MAX_PHASE1_FIXED_POINT_ITERS):
        iter_changed, split_candidates = _phase1_reorder(module, schedule, module_name)
        if not iter_changed:
            break
        changed = True
        schedule.update()
        schedule.verify()
        module.set_schedule(schedule)
    else:
        # Cap reached: the last update() was never rechecked (job 3188746: the
        # fill's own ENTRY total said ~59,760 us but the final schedule had
        # ~155,000 us exposed), so run one more pass.
        _logger.warning(
            "collective_overlap_pass [%s]: phase 1 fixed point not reached after %d iterations; "
            "running one final pass.",
            module_name, _MAX_PHASE1_FIXED_POINT_ITERS,
        )
        final_changed, split_candidates = _phase1_reorder(module, schedule, module_name)
        if final_changed:
            changed = True
            schedule.update()
            schedule.verify()
            module.set_schedule(schedule)
    return changed, split_candidates


def _compute_collective_overlap(serialized_hlo: bytes) -> Optional[bytes]:
    """Phase 1 (reorder) then phase 2 (split). Runs on rank 0 only; see
    _collective_overlap_pass."""
    if not _profile_costs:
        return None

    from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel
    module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
    schedule = module.schedule()
    if schedule is None:
        return None

    module_name = module.name

    _log_profile_coverage_gaps(module, schedule, module_name)

    changed, split_candidates = _run_phase1_to_fixed_point(module, schedule, module_name)
    _log_final_exposed_summary(module, schedule, module_name, "post-phase-1, pre-phase-2")

    phase1_bytes = module.as_serialized_hlo_module_proto() if changed else serialized_hlo

    if split_candidates:
        _logger.info(
            "collective_overlap_pass [%s]: %d split candidate(s) after phase 1.",
            module_name, len(split_candidates),
        )
        phase2_result = _phase2_split(phase1_bytes, split_candidates)
        if phase2_result is not None:
            phase2_bytes, group_latencies = phase2_result
            # The new sub-collectives have no PGLE entry; seed byte-weighted
            # estimates so the re-run doesn't skip them.
            _profile_costs.update(group_latencies)
            split_module = _hlo.HloModule.from_serialized_hlo_module_proto(phase2_bytes)
            split_schedule = split_module.schedule()
            if split_schedule is not None:
                split_changed, split_split_candidates = _run_phase1_to_fixed_point(
                    split_module, split_schedule, module_name
                )
                if split_split_candidates:
                    _logger.info(
                        "collective_overlap_pass [%s]: %d further split candidate(s) after re-optimizing; "
                        "not cascading into another split round.",
                        module_name, len(split_split_candidates),
                    )
                if split_changed:
                    phase2_bytes = split_module.as_serialized_hlo_module_proto()
                _log_final_exposed_summary(
                    split_module, split_schedule, module_name, "post-phase-2 split_module"
                )
            return phase2_bytes

    if changed:
        _log_final_exposed_summary(module, schedule, module_name, "phase-1-only module")
        return phase1_bytes
    return None


def _collective_overlap_pass(serialized_hlo: bytes) -> Optional[bytes]:
    """Run the pass on rank 0 and share its output bytes with every rank.

    Every rank must compile a byte-identical executable, so only rank 0
    computes and everyone returns the same bytes, through the coordination
    service key-value store.

    The KV key and barrier names use a per-process invocation counter, not a
    content hash. Hashing the serialized proto and then module.to_string() both
    failed: ranks' modules were provably equivalent yet hashed differently (a
    per-compile unique-id field in the proto; JSON key order inside an opaque
    cuDNN backend_config), leaving a rank waiting forever on a key rank 0 never
    wrote. A call-order key only needs every rank to reach the invocations in
    the same order, which has held in every run.

    The entry/exit barriers (raw coordination-service RPCs, safe inside a
    compiler callback unlike multihost_utils helpers, which compile and run a
    jitted collective) stop rank 0 racing ahead of the other ranks, which sit
    blocked in the KV read. Otherwise a later rank-0-only compile that eagerly
    runs a real cross-rank NCCL collective (observed: the MoE EP "borrowed comm"
    bootstrap creating the world clique) would deadlock against ranks frozen
    waiting on rank 0.
    """
    global _barrier_invocation_count

    # First statement, as cheap as possible, so the log shows the callback was
    # entered even if a hang happens later in XLA's C++ compiler.
    _t_enter = time.monotonic()
    _logger.info(
        "collective_overlap_pass: ENTER host=%s pid=%d input_bytes=%d",
        socket.gethostname(), os.getpid(), len(serialized_hlo),
    )

    # Early-out before touching the distributed client: modules with no async op
    # (jit_broadcast_in_dim etc.) need no synchronization at all.
    from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel
    module = _hlo.HloModule.from_serialized_hlo_module_proto(serialized_hlo)
    if not _module_has_interesting_async_ops(module):
        _logger.info(
            "collective_overlap_pass: EXIT host=%s pid=%d module=%s result=no-op (no async ops; %.1f ms).",
            socket.gethostname(), os.getpid(), module.name,
            (time.monotonic() - _t_enter) * 1000,
        )
        return None

    client = _get_distributed_client()
    _in_rank = _jax_process_id() if client is not None else 0
    dump_modules = os.environ.get("COLLECTIVE_OVERLAP_DUMP_MODULE", "1") == "1"

    if dump_modules:
        # Every rank's raw input, so ranks' modules can be diffed.
        _module_text = module.to_string()
        _in_host = socket.gethostname()
        _in_pid = os.getpid()
        _content_digest = hashlib.sha256(_module_text.encode()).hexdigest()
        _logger.info(
            "collective_overlap_pass: PRE-PASS INPUT HASH rank=%d host=%s pid=%d module=%s sha256=%s bytes=%d text_bytes=%d",
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

    _entry_name = _exit_name = None
    if multi_process:
        _barrier_invocation_count += 1
        _entry_name = f"collective_overlap_pass_entry_{_barrier_invocation_count}"
        _exit_name = f"collective_overlap_pass_exit_{_barrier_invocation_count}"
        try:
            client.wait_at_barrier(_entry_name, _SHARE_TIMEOUT_MS)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: entry barrier %s failed (%s); proceeding without it.",
                _entry_name, exc,
            )

    result_bytes: Optional[bytes] = None
    if is_root:
        _t_compute = time.monotonic()
        result_bytes = _compute_collective_overlap(serialized_hlo)
        _logger.info(
            "collective_overlap_pass: _compute_collective_overlap done (%s, %.1f ms).",
            "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
            (time.monotonic() - _t_compute) * 1000,
        )

    if multi_process:
        _key = f"collective_overlap_pass_kv_{_barrier_invocation_count}"
        try:
            if is_root:
                _payload = _SHARE_NONE if result_bytes is None else _SHARE_SOME + result_bytes
                client.key_value_set_bytes(_key, _payload)
            else:
                _payload = client.blocking_key_value_get_bytes(_key, _SHARE_TIMEOUT_MS)
                result_bytes = None if _payload == _SHARE_NONE else _payload[len(_SHARE_SOME):]
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: sharing the scheduled module via the JAX distributed client "
                "failed (%s); using this rank's own result, which risks divergent schedules.", exc,
            )
            if not is_root:
                result_bytes = _compute_collective_overlap(serialized_hlo)

        try:
            client.wait_at_barrier(_exit_name, _SHARE_TIMEOUT_MS)
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: exit barrier %s failed (%s); proceeding without it.",
                _exit_name, exc,
            )

    # What this rank returns, post-share: `grep "POST-SHARE HASH"` across rank
    # logs should show one hash per module name if the share worked.
    if result_bytes is not None and dump_modules:
        _rank = _jax_process_id()
        _host = socket.gethostname()
        _pid = os.getpid()
        _digest = hashlib.sha256(result_bytes).hexdigest()
        _final_module = _hlo.HloModule.from_serialized_hlo_module_proto(result_bytes)
        _module_name = _final_module.name
        _logger.info(
            "collective_overlap_pass: POST-SHARE HASH rank=%d host=%s pid=%d module=%s sha256=%s bytes=%d",
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
        "collective_overlap_pass: EXIT host=%s pid=%d result=%s (total %.1f ms).",
        socket.gethostname(), os.getpid(),
        "no-op" if result_bytes is None else f"{len(result_bytes)} bytes",
        (time.monotonic() - _t_enter) * 1000,
    )
    return result_bytes


# ---------------------------------------------------------------------------
# PGLE profile interception
# ---------------------------------------------------------------------------
_patched = False


def _patch_pgle_profiler() -> None:
    """Hook PGLE to capture its profile and the raw XSpace behind it."""
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

    # PGLEProfiler.trace() calls _profiler.get_fdo_profile(xspace) on each
    # retry's raw XSpace and then discards it. Wrapping it recovers the
    # per-invocation costs live, from this run's own profiling data.
    _original_get_fdo_profile = _jax_profiler._profiler.get_fdo_profile

    def _get_fdo_profile_and_capture(xspace):
        try:
            all_costs = _load_costs_from_xspace_bytes(
                bytes(xspace),
                (_TE_EP_PREFIX,) + _GEMM_COST_PREFIXES + _COLLECTIVE_COST_PREFIXES,
            )
            te_ep_costs = {k: v for k, v in all_costs.items() if k.startswith(_TE_EP_PREFIX)}
            gemm_costs = {k: v for k, v in all_costs.items() if k.startswith(_GEMM_COST_PREFIXES)}
            coll_costs = {k: v for k, v in all_costs.items() if k.startswith(_COLLECTIVE_COST_PREFIXES)}
            _gemm_overrides.update(gemm_costs)
            _collective_overrides.update(coll_costs)
            _te_ep_overrides.update(te_ep_costs)
            _profile_costs.update(te_ep_costs)
            _logger.info(
                "collective_overlap_pass: captured live per-invocation corrections: %d te_ep, %d GEMM, %d collective.",
                len(te_ep_costs), len(gemm_costs), len(coll_costs),
            )
        except Exception as exc:  # pylint: disable=broad-except
            _logger.warning(
                "collective_overlap_pass: failed to extract per-invocation costs from live PGLE data: %s", exc,
            )
        return _original_get_fdo_profile(xspace)

    _jax_profiler._profiler.get_fdo_profile = _get_fdo_profile_and_capture


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def register() -> None:
    """Register the collective-overlap POST_SCHEDULER pass and PGLE hook."""
    # A hang inside XLA's C++ compiler leaves no log lines and can't be reached
    # from outside the container; `kill -USR1 <pid>` dumps every Python thread's
    # traceback to stderr instead.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
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
    _logger.info("collective_overlap_pass: registered POST_SCHEDULER pass 'profile_guided_collective_overlap'.")


# ---------------------------------------------------------------------------
# Subprocess entry point for phase 2 (avoids descriptor pool conflicts)
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
