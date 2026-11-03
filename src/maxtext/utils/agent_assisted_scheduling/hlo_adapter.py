"""Adapter between `jax._src.lib.hlo.HloModule` and the jax-free schedule model.

`load_module` reads a scheduled module plus an FDO `Profile` into
`schedule_model.Module` and the baseline order of every computation. `apply_plan`
validates a plan against it and, only if it is fully legal, writes it back and
lets XLA re-check it, restoring the baseline on any problem.

The binding objects are used through duck typing (module.schedule(),
module.make_nonfusion_computations(), computation.instructions(), instruction
name/opcode/operands/to_string/async_wrapped_root, schedule.sequence/
set_sequence/update/verify), so a fake with that surface works in tests.

Approximations, all conservative:
  * Costs follow `gpu_collective_overlap_pass._resolve_inst_cost`: the instruction's own
    profile entry, else (for a kCustom fusion) the sum of its nested entries;
    unprofiled ops cost 0. Starts and dones cost 0 (their latency is separate).
  * A start's latency is keyed like the pass does: the wrapped root, unwrapped
    through call/async wrappers.
  * Host-offload copies are recognised by an async start whose text touches
    host memory space S(5); they cost 0, are not on the comm stream, and a D2H
    (dynamic-update-slice root) is marked as writing its first operand.
  * Device groups that cannot be parsed are left as None (unknown).
"""

# Binding calls can raise arbitrary errors for attributes a given XLA build does not
# expose; loading degrades to "attribute unknown" and apply_plan fails closed, so broad
# excepts are intentional here.
# pylint: disable=broad-exception-caught

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from maxtext.utils.agent_assisted_scheduling import hlo_text
from maxtext.utils.agent_assisted_scheduling.legality import Severity, Violation, errors, validate_plan
from maxtext.utils.agent_assisted_scheduling.plan import IdentityOptions, Plan, verify_against
from maxtext.utils.agent_assisted_scheduling.profile import Profile
from maxtext.utils.agent_assisted_scheduling.schedule_model import (
    AsyncRole,
    Computation,
    Inst,
    Module,
    TRIVIAL_OPCODES,
)

_COLLECTIVE_PREFIXES = ("all-gather", "reduce-scatter", "all-reduce")
_MAX_UNWRAP_HOPS = 8

# Opcodes allowed inside a fused body for the fusion to still count as trivially movable.
# Mirrors _TRIVIAL_FUSED_OPCODES in gpu_collective_overlap_pass.py.
_TRIVIAL_FUSED_OPCODES = frozenset(
    {
        "parameter",
        "constant",
        "iota",
        "convert",
        "bitcast",
        "reshape",
        "transpose",
        "broadcast",
        "copy",
        "concatenate",
        "dynamic-slice",
        "slice",
        "get-tuple-element",
        "tuple",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "abs",
        "maximum",
        "minimum",
        "and",
        "or",
        "not",
        "xor",
        "sqrt",
        "rsqrt",
        "exp",
        "log",
        "sign",
        "clamp",
        "floor",
        "ceil",
        "round-nearest-afz",
        "round-nearest-even",
        "compare",
        "select",
    }
)


def _opcode(inst) -> str:
  return hlo_text.opcode_str(inst.opcode.name)


def _text(inst) -> str:
  try:
    return inst.to_string()
  except Exception:  # binding errors must not abort loading; treat as no attributes
    return ""


def _is_start(opcode: str) -> bool:
  return opcode.endswith("-start") or opcode == "async-start"


def _is_done(opcode: str) -> bool:
  return opcode.endswith("-done") or opcode == "async-done"


@dataclass
class LoadedModule:
  """A module in the library's model plus handles back into the binding."""

  module: Module
  baseline: dict[str, list[str]]
  while_bodies: frozenset[str] = frozenset()
  # computation name -> instruction name -> binding instruction
  handles: dict[str, dict[str, Any]] = field(repr=False, default_factory=dict)
  hlo_module: Any = field(repr=False, default=None)
  profile: Profile = field(repr=False, default_factory=Profile)


class _Loader:
  """Per-module state for turning binding instructions into `Inst`s."""

  def __init__(self, hlo_module, profile: Profile, sm_gate_ops: str = "gemm"):
    self.profile = profile
    self.sm_gate_ops = sm_gate_ops
    self.comp_by_name = {c.name: c for c in hlo_module.computations()}

  def _callee(self, inst):
    name = hlo_text.callee_name(_text(inst))
    return self.comp_by_name.get(name) if name else None

  def _root(self, comp):
    try:
      return comp.root_instruction()
    except Exception:
      pass
    # Binding without root_instruction(): the unique instruction with no users.
    roots = [i for i in comp.instructions() if not list(i.users())]
    return roots[-1] if roots else None

  def _unwrap_inst(self, inst):
    for _ in range(_MAX_UNWRAP_HOPS):
      if _opcode(inst) not in ("call", "call-start", "async-start"):
        return inst
      callee = self._callee(inst)
      root = self._root(callee) if callee is not None else None
      if root is None:
        return inst
      inst = root
    return inst

  def _unwrap(self, inst) -> str:
    return self._unwrap_inst(inst).name

  def _wrapped_leaf(self, start):
    """The instruction a start wraps (its async computation's root, unwrapped), or None."""
    root = None
    try:
      root = start.async_wrapped_root()
    except Exception:
      root = None
    if root is None:
      callee = self._callee(start)
      root = self._root(callee) if callee is not None else None
    return None if root is None else self._unwrap_inst(root)

  def profile_key(self, start) -> str:
    """Name under which `start`'s latency is keyed in the profile."""
    leaf = self._wrapped_leaf(start)
    return start.name if leaf is None else leaf.name

  def wrapped_opcode(self, start) -> Optional[str]:
    """Opcode of the wrapped op ("all-gather", "custom-call", ...), None if unknown."""
    leaf = self._wrapped_leaf(start)
    return None if leaf is None else _opcode(leaf)

  def _fused_comp(self, inst):
    return self._callee(inst) if _opcode(inst) == "fusion" else None

  def cost_us(self, inst, host_copy: bool) -> float:
    if host_copy:
      return 0.0
    direct = self.profile.cost(inst.name)
    if direct:
      return direct
    if _opcode(inst) != "fusion" or "kind=kCustom" not in _text(inst):
      return 0.0
    comp = self._fused_comp(inst)
    return sum(self.profile.cost(i.name) for i in comp.instructions()) if comp is not None else 0.0

  def trivially_movable(self, inst) -> bool:
    opcode = _opcode(inst)
    if opcode in TRIVIAL_OPCODES:
      return True
    if opcode == "fusion":
      comp = self._fused_comp(inst)
      return comp is not None and all(_opcode(i) in _TRIVIAL_FUSED_OPCODES for i in comp.instructions())
    return False

  def _is_sm_filling(self, inst) -> bool:
    """dot/convolution, or a custom call of a GEMM/convolution/attention kernel family."""
    opcode = _opcode(inst)
    if opcode in ("dot", "convolution"):
      return True
    return opcode == "custom-call" and bool(hlo_text.GEMM_LIKE_TARGET_RE.search(_text(inst)))

  def holds_sms(self, inst) -> bool:
    """Whether `inst` can keep a collective from starting while it runs.

    sm_gate_ops "gemm" (default): dot/convolution, GEMM-like custom calls, and
    fusions wrapping either. "all": any non-trivial compute op (the simulator
    still ignores ones cheaper than the gate threshold). "none": never.
    """
    mode = self.sm_gate_ops
    if mode == "none":
      return False
    if mode == "all":
      return not self.trivially_movable(inst)
    if self._is_sm_filling(inst):
      return True
    if _opcode(inst) == "fusion":
      comp = self._fused_comp(inst)
      return comp is not None and any(self._is_sm_filling(i) for i in comp.instructions())
    return False

  def host_copy(self, start) -> tuple[bool, bool]:
    """(is a host-offload copy start, is a D2H write into the host buffer)."""
    text = _text(start)
    callee = self._callee(start)
    body = text + (" ".join(_text(i) for i in callee.instructions()) if callee is not None else "")
    if not hlo_text.touches_host_memory(body):
      return False, False
    root = self._root(callee) if callee is not None else None
    return True, root is not None and _opcode(root) == "dynamic-update-slice"

  def load_inst(self, inst, host_starts: set[str]) -> Inst:
    """One binding instruction as an `Inst`."""
    opcode = _opcode(inst)
    name = inst.name
    text = _text(inst)
    operands = tuple(o.name for o in inst.operands())
    role = AsyncRole.START if _is_start(opcode) else AsyncRole.DONE if _is_done(opcode) else AsyncRole.NONE
    is_host_start = name in host_starts
    kwargs: dict[str, Any] = {
        "name": name,
        "opcode": opcode,
        "operands": operands,
        "control_predecessors": hlo_text.control_predecessor_names(text),
        "trivially_movable": self.trivially_movable(inst),
    }
    if role is AsyncRole.NONE:
      kwargs["cost_us"] = self.cost_us(inst, False)
      kwargs["sm_gate"] = self.holds_sms(inst)
    else:
      kwargs["async_role"] = role
    if role is AsyncRole.START and not is_host_start:
      key = self.profile_key(inst)
      kwargs["profile_key"] = key
      kwargs["wrapped_opcode"] = self.wrapped_opcode(inst)
      kwargs["latency_us"] = self.profile.cost(key) or None
      kwargs["uses_comm_stream"] = True
      kwargs["device_groups"] = hlo_text.parse_device_groups(text)
    if is_host_start:
      _, writes = self.host_copy(inst)
      if writes and operands:
        kwargs["writes_operand"] = operands[0]
    return Inst(**kwargs)


SM_GATE_OPS = ("gemm", "all", "none")


def load_module(hlo_module, profile: Optional[Profile] = None, sm_gate_ops: str = "gemm") -> LoadedModule:
  """Read a scheduled HloModule and an FDO profile into the library model.

  `sm_gate_ops` picks which ops hold the SMs in the comm-stream simulation (see
  `_Loader.holds_sms`): "gemm" (default), "all" or "none".

  Raises ValueError if the module has no schedule (nothing to reorder) or the
  schedule and the computation's instruction set disagree.
  """
  if sm_gate_ops not in SM_GATE_OPS:
    raise ValueError(f"sm_gate_ops must be one of {SM_GATE_OPS}, not {sm_gate_ops!r}")
  profile = profile or Profile()
  schedule = hlo_module.schedule()
  if schedule is None:
    raise ValueError(f"module {hlo_module.name} is not scheduled")
  loader = _Loader(hlo_module, profile, sm_gate_ops)
  comps: list[Computation] = []
  baseline: dict[str, list[str]] = {}
  handles: dict[str, dict[str, Any]] = {}
  while_bodies: set[str] = set()
  for hcomp in hlo_module.make_nonfusion_computations():
    seq = list(schedule.sequence(hcomp))
    by_name = {i.name: i for i in hcomp.instructions()}
    if set(by_name) != {i.name for i in seq} or len(by_name) != len(seq):
      raise ValueError(f"schedule of {hcomp.name} does not match its instructions")
    host_starts = {i.name for i in seq if _opcode(i) == "async-start" and loader.host_copy(i)[0]}
    for i in seq:
      if _opcode(i) == "while":
        body = hlo_text.while_body_name(_text(i))
        if body:
          while_bodies.add(body)
    comps.append(Computation.from_instructions(hcomp.name, [loader.load_inst(i, host_starts) for i in seq]))
    baseline[hcomp.name] = [i.name for i in seq]
    handles[hcomp.name] = by_name
  return LoadedModule(
      Module.from_computations(hlo_module.name, comps),
      baseline,
      frozenset(while_bodies),
      handles,
      hlo_module,
      profile=profile,
  )


@dataclass(frozen=True)
class ApplyResult:
  applied: bool
  violations: tuple[Violation, ...] = ()
  reason: str = ""


def apply_plan(
    loaded: LoadedModule,
    plan: "Plan | dict[str, list[str]]",
    *,
    check_device_groups: bool = False,
    identity: IdentityOptions = IdentityOptions(),
) -> ApplyResult:
  """Validate `plan` and, if fully legal, write it into the module.

  A `Plan` is first matched against this module (name, checksum of XLA's input
  schedule, profile identity); a mismatch rejects it. Pass
  `IdentityOptions(require_identity=True)` to also refuse unstamped plans, which
  is what a production pass should do.

  Fail closed: any ERROR violation (including a cross-computation move) leaves
  the module untouched. `module.schedule()` returns a copy, so the plan is
  applied to that copy, checked by XLA's own `update()`/`verify()`, compared with
  what was planned (update can silently relocate an instruction whose dependency
  the plan broke), and only then committed with `module.set_schedule`. Any
  failure discards the copy. Computations whose planned order equals the
  baseline are not touched.
  """
  plan = plan if isinstance(plan, Plan) else Plan(computations=plan)
  issues = verify_against(plan, loaded.module, loaded.baseline, loaded.profile, identity)
  issues += validate_plan(
      loaded.module, plan.computations, baseline=loaded.baseline, check_device_groups=check_device_groups
  )
  if errors(issues):
    return ApplyResult(False, tuple(issues), "plan rejected by validator")

  changed = [c for c, order in plan.computations.items() if order != loaded.baseline[c]]
  if not changed:
    return ApplyResult(True, tuple(issues), "plan equals baseline; nothing to apply")

  def rejected(code: str, message: str) -> ApplyResult:
    return ApplyResult(False, tuple(issues) + (Violation(code, message),), message)

  hlo_module = loaded.hlo_module
  schedule = hlo_module.schedule()
  hcomps = {c.name: c for c in hlo_module.make_nonfusion_computations()}
  try:
    for cname in changed:
      schedule.set_sequence(hcomps[cname], [loaded.handles[cname][n] for n in plan.computations[cname]])
    schedule.update()
    schedule.verify()
  except Exception as exc:  # XLA's own verifier is the final authority
    return rejected("BINDING_REJECTED", f"XLA rejected the plan: {exc}")
  for cname in changed:
    if [i.name for i in schedule.sequence(hcomps[cname])] != plan.computations[cname]:
      return rejected(
          "UPDATE_RELOCATED",
          f"schedule.update() reordered {cname} away from the plan; a window scored as hidden may be exposed",
      )
  hlo_module.set_schedule(schedule)
  return ApplyResult(True, tuple(issues), "applied")


__all__ = ["ApplyResult", "LoadedModule", "Severity", "apply_plan", "load_module"]
