"""Priority-collective regression check.

A proposed order must not make a high-priority collective worse, even when the
total improves. Priority collectives are the ones whose wrapped op (all-gather,
reduce-scatter, all-reduce) starts with one of `DEFAULT_PRIORITY_PREFIXES` (the FSDP/TP all-gather, reduce-scatter and
all-reduce); other async ops (all-to-all, collective-permute, async custom calls)
share the comm stream but are not priority.

Three measures are compared between the baseline order and the proposed one, per
collective, and any increase is a regression:

  exposed    static deficit: latency minus the compute between start and done.
  stall      the stall of the dones in the one-FIFO comm-stream simulation, summed
             over the priority collectives: with one queue, stall moves between
             neighbouring dones, so a single done may rise while the others fall.
             The simulation can call a window hidden only because compute stalled
             on another collective, so the static deficit is checked as well.
  reachable  deficit if the done sank to its first real consumer (looking through
             get-tuple-element, bitcast and the like). Moving a consumer of a done
             earlier caps that sinking and shrinks the collective's overlap even
             though its current overlap is unchanged. Capped by the latency, so a
             window that stays fully hidable costs nothing.

All three apply to priority collectives only; the others yield.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from maxtext.utils.agent_assisted_scheduling.cost import prefix_costs
from maxtext.utils.agent_assisted_scheduling.schedule_model import AsyncRole, Computation, Inst, positions
from maxtext.utils.agent_assisted_scheduling.simulate import DEFAULT_SM_GATE_MIN_US, simulate_comm_stream

DEFAULT_PRIORITY_PREFIXES = ("all-gather", "reduce-scatter", "all-reduce")
_EPS_US = 1e-6


@dataclass(frozen=True)
class PriorityRegression:
  """One collective that is worse in the proposed order."""

  computation: str
  start: str
  kind: str  # "exposed" | "stall" | "reachable"
  before_us: float
  after_us: float

  @property
  def message(self) -> str:
    what = {
        "exposed": "exposed time (latency minus overlap)",
        "stall": "simulated comm-stream stall, summed",
        "reachable": "exposure if its done sank to its first real consumer",
    }[self.kind]
    return f"{self.start}: {what} rises {self.before_us:,.1f} -> {self.after_us:,.1f} us"


ALL_PRIORITY = "(all priority collectives)"


def is_priority(inst: Inst, prefixes: tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES) -> bool:
  """True for a start whose wrapped opcode (else profile key, else opcode) begins with a priority prefix.

  The wrapped opcode is the reliable signal (the profile key is only a name such
  as "all-gather.3.1"); the key is the fallback a profile-only analysis has.
  """
  kind = inst.wrapped_opcode or inst.profile_key or inst.opcode
  return inst.async_role is AsyncRole.START and kind.startswith(tuple(prefixes))


def real_consumers(comp: Computation, name: str) -> list[str]:
  """Users of `name` that are not trivially movable, looking through trivial ones."""
  users = comp.users()
  seen: set[str] = set()
  out: list[str] = []
  stack = list(users.get(name, ()))
  while stack:
    cur = stack.pop()
    if cur in seen:
      continue
    seen.add(cur)
    if comp[cur].is_trivially_movable:
      stack.extend(users.get(cur, ()))
    else:
      out.append(cur)
  return out


def reachable_overlap_us(comp: Computation, order: list[str], start: str, done: str) -> float:
  """Compute between `start` and the first real consumer of `done`, capped at the latency.

  The overlap the collective would have if its done sank as far as it can. With
  no consumer in the computation, the done may sink to the end.
  """
  latency = comp[start].latency_us or 0.0
  pos = positions(order)
  consumers = [pos[c] for c in real_consumers(comp, done) if c in pos]
  bound = min(consumers) if consumers else len(order)
  prefix = prefix_costs(comp, order)
  s = pos[start]
  return min(latency, prefix[bound] - prefix[s + 1]) if bound > s + 1 else 0.0


def collective_measures(
    comp: Computation,
    order: list[str],
    prefixes: tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES,
    *,
    sm_gate: bool = True,
    sm_gate_min_us: float = DEFAULT_SM_GATE_MIN_US,
) -> dict[str, dict[str, float]]:
  """{start: {"exposed", "reachable", "stall"}} for every profiled priority pair in `order`."""
  pos = positions(order)
  prefix = prefix_costs(comp, order)
  stalls = simulate_comm_stream(comp, order, sm_gate=sm_gate, sm_gate_min_us=sm_gate_min_us).stalls
  out: dict[str, dict[str, float]] = {}
  for start, done in comp.async_pairs():
    latency = comp[start].latency_us
    if not latency or latency <= 0 or start not in pos or done not in pos:
      continue
    if not is_priority(comp[start], prefixes):
      continue
    s, d = pos[start], pos[done]
    overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
    out[start] = {
        "exposed": max(0.0, latency - overlap),
        "reachable": max(0.0, latency - reachable_overlap_us(comp, order, start, done)),
        "stall": stalls.get(done, 0.0),
    }
  return out


def priority_exposed_us(comp: Computation, order: list[str], prefixes: tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES) -> float:
  """Summed static deficit of the priority collectives."""
  return sum(m["exposed"] for m in collective_measures(comp, order, prefixes).values())


def priority_regressions(
    comp: Computation,
    before: list[str],
    after: list[str],
    prefixes: tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES,
    *,
    sm_gate: bool = True,
    tol_us: float = _EPS_US,
) -> list[PriorityRegression]:
  """Collectives that are worse in `after` than in `before`, worst first.

  Collectives present in only one order are not compared.
  """
  b = collective_measures(comp, before, prefixes, sm_gate=sm_gate)
  a = collective_measures(comp, after, prefixes, sm_gate=sm_gate)
  out: list[PriorityRegression] = []
  for start, after_m in a.items():
    before_m: Optional[dict[str, float]] = b.get(start)
    if before_m is None:
      continue
    for kind in ("exposed", "reachable"):
      if after_m[kind] > before_m[kind] + tol_us:
        out.append(PriorityRegression(comp.name, start, kind, before_m[kind], after_m[kind]))
  both = [s for s in a if s in b]
  stall_before = sum(b[s]["stall"] for s in both)
  stall_after = sum(a[s]["stall"] for s in both)
  if stall_after > stall_before + tol_us:
    out.append(PriorityRegression(comp.name, ALL_PRIORITY, "stall", stall_before, stall_after))
  return sorted(out, key=lambda r: -(r.after_us - r.before_us))
