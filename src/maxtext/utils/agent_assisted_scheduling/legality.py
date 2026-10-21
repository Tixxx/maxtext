"""Deterministic legality checks for a proposed schedule of one computation.

Encodes the checklist in skills/xla-hlo-schedule-legality. Errors mean the
proposal must be rejected whole; warnings flag a legal but self-defeating order.

Instructions never move between computations: `validate_plan` rejects any plan
that puts an instruction in a computation other than its own (CROSS_COMPUTATION).

What this does NOT check: memory, aliasing hazards that are not already
operand/control edges or `writes_operand`, or scheduling-annotation groups. The final authority is still
`HloSchedule.update()/verify()` inside XLA.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

from maxtext.utils.agent_assisted_scheduling.schedule_model import (
    AsyncRole,
    Computation,
    Module,
    NOP_OPCODES,
    positions,
)


class Severity(enum.Enum):
  ERROR = "error"
  WARNING = "warning"


@dataclass(frozen=True)
class Violation:
  code: str
  message: str
  instruction: Optional[str] = None
  severity: Severity = Severity.ERROR


def _structure_violations(comp: Computation, order: list[str]) -> list[Violation]:
  """Exactly-once membership, mirroring HloSchedule::Verify."""
  out: list[Violation] = []
  seen: set[str] = set()
  for name in order:
    if name in seen:
      out.append(Violation("DUPLICATE", f"{name} appears more than once", name))
    seen.add(name)
    if name not in comp:
      out.append(Violation("UNKNOWN", f"{name} is not in computation {comp.name}", name))
  for name in comp.instructions:
    if name not in seen:
      out.append(Violation("MISSING", f"{name} is not in the schedule", name))
  return out


def _edge_violations(comp: Computation, pos: dict[str, int]) -> list[Violation]:
  """Operand and control-predecessor order, mirroring HloSchedule::Verify."""
  out: list[Violation] = []
  for inst in comp.instructions.values():
    if inst.name not in pos:
      continue
    for op in inst.operands:
      if op in pos and pos[op] >= pos[inst.name]:
        out.append(Violation("OPERAND_ORDER", f"{inst.name} is not scheduled after its operand {op}", inst.name))
    for pred in inst.control_predecessors:
      if pred in pos and pos[pred] >= pos[inst.name]:
        out.append(
            Violation("CONTROL_ORDER", f"{inst.name} is not scheduled after its control predecessor {pred}", inst.name)
        )
  return out


def _async_violations(comp: Computation) -> list[Violation]:
  """Start/done pairing, mirroring the HLO verifier's async rules."""
  out: list[Violation] = []
  users = comp.users()
  dones_seen: dict[str, str] = {}
  for inst in comp.instructions.values():
    if inst.async_role is AsyncRole.DONE:
      if len(inst.operands) != 1:
        out.append(Violation("ASYNC_PAIR", f"done {inst.name} must have exactly one operand", inst.name))
        continue
      start = inst.operands[0]
      if start not in comp or comp[start].async_role is not AsyncRole.START:
        out.append(Violation("ASYNC_PAIR", f"done {inst.name} does not trace back to an async start", inst.name))
      elif start in dones_seen:
        out.append(
            Violation("ASYNC_PAIR", f"start {start} has multiple dones: {dones_seen[start]} and {inst.name}", start)
        )
      else:
        dones_seen[start] = inst.name
  for inst in comp.instructions.values():
    if inst.async_role is AsyncRole.START and inst.name not in dones_seen:
      out.append(Violation("ASYNC_PAIR", f"start {inst.name} has no matching done", inst.name))
    if inst.async_role is AsyncRole.START:
      extra = [
          u for u in users[inst.name] if comp[u].async_role is not AsyncRole.DONE and comp[u].opcode not in NOP_OPCODES
      ]
      if extra:
        out.append(Violation("ASYNC_PAIR", f"start {inst.name} is used by non-done instruction(s) {extra}", inst.name))
  return out


def _sync_conversion_warnings(comp: Computation, order: list[str]) -> list[Violation]:
  """A start with only no-ops before its done becomes a synchronous collective."""
  out: list[Violation] = []
  pos = positions(order)
  for start, done in comp.async_pairs():
    if start not in pos or done not in pos or pos[start] >= pos[done]:
      continue
    between = order[pos[start] + 1 : pos[done]]
    if all(n in comp and comp[n].opcode in NOP_OPCODES for n in between):
      out.append(
          Violation(
              "SYNC_CONVERSION",
              f"{start}..{done} has no real op in between; XLA will make it synchronous, " "so it hides nothing",
              start,
              Severity.WARNING,
          )
      )
  return out


def _devices(groups) -> set[int]:
  return set().union(*groups) if groups else set()


def _overlap_violations(
    comp: Computation,
    order: list[str],
    check_device_groups: bool,
) -> list[Violation]:
  """Concurrent comm-stream collectives vs participating device groups.

  There is deliberately no cap on how many collectives are in flight: XLA's
  overlap limit is a scheduler heuristic, not a correctness rule.
  """
  out: list[Violation] = []
  inflight: list[str] = []
  for name in order:
    if name not in comp:
      continue
    inst = comp[name]
    if inst.async_role is AsyncRole.START and inst.uses_comm_stream:
      if check_device_groups:
        for other in inflight:
          a, b = inst.device_groups, comp[other].device_groups
          if not a or not b:
            out.append(
                Violation(
                    "DEVICE_GROUPS_UNKNOWN", f"{name} overlaps {other} but participating device groups are unknown", name
                )
            )
            continue
          shared = max(len(g1 & g2) for g1 in a for g2 in b)
          if shared > 1:
            out.append(
                Violation(
                    "DEVICE_GROUP_OVERLAP",
                    f"{name} overlaps in-flight {other} sharing {shared} devices " "(cyclic launch dependency)",
                    name,
                )
            )
      inflight.append(name)
    elif inst.async_role is AsyncRole.DONE and inst.operands and inst.operands[0] in inflight:
      inflight.remove(inst.operands[0])
  return out


def _in_place_violations(
    comp: Computation,
    baseline_pos: dict[str, int],
    pos: dict[str, int],
) -> list[Violation]:
  """Users of an in-place-written buffer keep their baseline order with the writer."""
  out: list[Violation] = []
  users = comp.users()
  for writer in comp.instructions.values():
    buf = writer.writes_operand
    if buf is None or writer.name not in pos or writer.name not in baseline_pos:
      continue
    for other in users.get(buf, []):
      if other == writer.name or other not in pos or other not in baseline_pos:
        continue
      was_before = baseline_pos[other] < baseline_pos[writer.name]
      now_before = pos[other] < pos[writer.name]
      if was_before != now_before:
        out.append(
            Violation(
                "IN_PLACE_REORDER",
                f"{other} and in-place writer {writer.name} swapped order on buffer {buf}",
                writer.name,
            )
        )
  return out


def validate_order(
    comp: Computation,
    order: list[str],
    *,
    baseline: Optional[list[str]] = None,
    check_device_groups: bool = False,
) -> list[Violation]:
  """All violations of `order` as a schedule for `comp` (empty list = accept).

  Args:
    comp: the computation (instruction set and edges).
    order: proposed instruction names, first to last.
    baseline: the schedule XLA produced; enables the in-place reorder check.
    check_device_groups: also reject concurrent collectives sharing > 1 device
      (GpuScheduleCrossesOverlapLimit); unknown groups are reported as errors.
      Enable only when xla_gpu_experimental_enable_collective_multi_streaming is
      on or the module uses multiple collective domains; otherwise XLA has no
      such constraint.
  """
  out = _structure_violations(comp, order)
  if out:  # position-based checks are meaningless on a malformed order
    return out
  pos = positions(order)
  out += _edge_violations(comp, pos)
  out += _async_violations(comp)
  out += _overlap_violations(comp, order, check_device_groups)
  if baseline is not None:
    base_issues = _structure_violations(comp, baseline)
    if base_issues:
      out.append(Violation("BAD_BASELINE", "baseline is not a valid order for this computation"))
    else:
      out += _in_place_violations(comp, positions(baseline), pos)
  out += _sync_conversion_warnings(comp, order)
  return out


def validate_plan(
    module: Module,
    plan: dict[str, list[str]],
    *,
    baseline: Optional[dict[str, list[str]]] = None,
    check_device_groups: bool = False,
) -> list[Violation]:
  """All violations of a whole-module plan (empty list = accept).

  `plan` maps computation name to its full proposed order. Computations absent
  from the plan keep their baseline order and are not checked. Strictly no
  cross-computation moves: every name in `plan[c]` must be an instruction of
  computation `c`. Moving X out of A shows up as MISSING in A and as
  CROSS_COMPUTATION in wherever it was placed; a plan naming a computation the
  module does not have is PLAN_UNKNOWN_COMPUTATION. Everything else is
  `validate_order` per computation.
  """
  out: list[Violation] = []
  for cname, order in plan.items():
    comp = module.computations.get(cname)
    if comp is None:
      out.append(
          Violation("PLAN_UNKNOWN_COMPUTATION", f"plan names computation {cname!r}, which is not in module {module.name}")
      )
      continue
    foreign = {}
    for name in order:
      owner = module.owner.get(name)
      if owner is not None and owner != cname:
        foreign[name] = owner
        out.append(
            Violation(
                "CROSS_COMPUTATION",
                f"{name} belongs to computation {owner!r} but the plan places it in {cname!r}; "
                "instructions must stay in their own computation",
                name,
            )
        )
    inner = validate_order(comp, order, baseline=(baseline or {}).get(cname), check_device_groups=check_device_groups)
    out += [v for v in inner if not (v.code == "UNKNOWN" and v.instruction in foreign)]
  return out


def errors(violations: list[Violation]) -> list[Violation]:
  return [v for v in violations if v.severity is Severity.ERROR]


def is_legal(comp: Computation, order: list[str], **kwargs) -> bool:
  """True if `order` has no ERROR-level violations (warnings allowed)."""
  return not errors(validate_order(comp, order, **kwargs))
