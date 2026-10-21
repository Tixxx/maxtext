"""Ranked exposed-collective report and before/after plan comparison.

jax-free: works on `schedule_model.Module` and name-list schedules, so the same
code serves the CLI, tests and a future plan-applying pass. Exposure uses the
static window metric from `cost.py` (latency minus the summed cost between start
and done); `sim_stall_us` adds the one-FIFO-stream, SM-gated simulation from
`simulate.py`. Instructions without a profile entry count as free, so every row
says whether its latency was profiled and each computation lists coverage gaps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from maxtext.utils.agent_assisted_scheduling.cost import coverage_gaps, total_exposed_us
from maxtext.utils.agent_assisted_scheduling.legality import Violation, errors, validate_plan
from maxtext.utils.agent_assisted_scheduling.plan import IdentityOptions, Plan, verify_against
from maxtext.utils.agent_assisted_scheduling.placement import earliest_legal_pos, hoist_start_to_floor
from maxtext.utils.agent_assisted_scheduling.schedule_model import Computation, Module, positions
from maxtext.utils.agent_assisted_scheduling.profile import Profile
from maxtext.utils.agent_assisted_scheduling.simulate import simulate_comm_stream


@dataclass(frozen=True)
class CollectiveRow:
  """One async start/done pair."""

  start: str
  done: str
  key: Optional[str]
  profiled: bool
  latency_us: Optional[float]
  overlap_us: float
  deficit_us: float
  sim_stall_us: float
  start_pos: int
  done_pos: int
  floor: int
  hoist_to: Optional[int]
  blocker: Optional[str]
  control_pins: tuple[str, ...]

  @property
  def can_hoist(self) -> bool:
    """True if hoisting the start to its floor would move it earlier."""
    return self.hoist_to is not None


@dataclass(frozen=True)
class ComputationReport:
  """Exposure of one computation under one order."""

  name: str
  while_body: bool
  num_instructions: int
  static_exposed_us: float
  sim_stall_us: float
  rows: tuple[CollectiveRow, ...]
  gaps: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModuleReport:
  """Exposure of every computation that has an async pair."""

  module: str
  computations: tuple[ComputationReport, ...]

  @property
  def static_exposed_us(self) -> float:
    return sum(c.static_exposed_us for c in self.computations)

  @property
  def sim_stall_us(self) -> float:
    return sum(c.sim_stall_us for c in self.computations)

  def ranked(self, min_deficit_us: float = 0.0) -> list[tuple[str, CollectiveRow]]:
    """(computation, row) pairs, profiled rows with a deficit first, largest first.

    Unprofiled rows sort last: their exposure is unknown, not zero.
    """
    pairs = [(c.name, r) for c in self.computations for r in c.rows]
    pairs = [p for p in pairs if not p[1].profiled or p[1].deficit_us >= min_deficit_us]
    return sorted(pairs, key=lambda p: (not p[1].profiled, -p[1].deficit_us, -p[1].sim_stall_us, p[1].start))


def _row(comp: Computation, order: list[str], start: str, done: str, stalls: dict[str, float]) -> CollectiveRow:
  pos = positions(order)
  inst = comp[start]
  latency = inst.latency_us
  profiled = bool(latency and latency > 0)
  overlap = sum(comp[n].cost_us for n in order[pos[start] + 1 : pos[done]])
  elp = earliest_legal_pos(comp, order, start)
  hoisted = hoist_start_to_floor(comp, order, start)
  new_pos = hoisted.index(start)
  return CollectiveRow(
      start=start,
      done=done,
      key=inst.profile_key,
      profiled=profiled,
      latency_us=latency if profiled else None,
      overlap_us=overlap,
      deficit_us=max(0.0, latency - overlap) if profiled else 0.0,
      sim_stall_us=stalls.get(done, 0.0),
      start_pos=pos[start],
      done_pos=pos[done],
      floor=elp.floor,
      hoist_to=new_pos if new_pos < pos[start] else None,
      blocker=elp.blocker,
      control_pins=tuple(name for _, name in elp.control_pins),
  )


def analyze_computation(
    comp: Computation, order: list[str], *, while_body: bool = False, sm_gate: bool = True
) -> ComputationReport:
  """Exposure report of `comp` scheduled as `order`."""
  sim = simulate_comm_stream(comp, order, sm_gate=sm_gate)
  rows = tuple(_row(comp, order, s, d, sim.stalls) for s, d in comp.async_pairs())
  return ComputationReport(
      name=comp.name,
      while_body=while_body,
      num_instructions=len(order),
      static_exposed_us=total_exposed_us(comp, order),
      sim_stall_us=sim.total_stall_us,
      rows=rows,
      gaps=tuple(coverage_gaps(comp, order)),
  )


def analyze_module(
    module: Module,
    orders: dict[str, list[str]],
    *,
    while_bodies: frozenset[str] = frozenset(),
    only: Optional[set[str]] = None,
    sm_gate: bool = True,
) -> ModuleReport:
  """Report every computation with an async pair (restricted to `only` if given)."""
  reports = []
  for name, comp in module.computations.items():
    if only is not None and name not in only:
      continue
    if not comp.async_pairs():
      continue
    reports.append(analyze_computation(comp, orders[name], while_body=name in while_bodies, sm_gate=sm_gate))
  return ModuleReport(module.name, tuple(reports))


@dataclass(frozen=True)
class ComputationDelta:
  """Before/after totals for one computation touched by a plan."""

  name: str
  static_before_us: float
  static_after_us: float
  sim_before_us: float
  sim_after_us: float


@dataclass(frozen=True)
class PlanReport:
  """Outcome of checking a plan; `after` is None when the plan has errors."""

  violations: tuple[Violation, ...]
  before: ModuleReport
  after: Optional[ModuleReport] = None
  deltas: tuple[ComputationDelta, ...] = field(default_factory=tuple)

  @property
  def accepted(self) -> bool:
    return not errors(list(self.violations))


def check_plan(
    module: Module,
    baseline: dict[str, list[str]],
    plan: "Plan | dict[str, list[str]]",
    *,
    profile: Optional[Profile] = None,
    identity: IdentityOptions = IdentityOptions(),
    while_bodies: frozenset[str] = frozenset(),
    check_device_groups: bool = False,
    sm_gate: bool = True,
) -> PlanReport:
  """Validate `plan` and, if legal, compare exposure before and after.

  `plan` is a `Plan` (its module/checksum/profile identity is checked first) or a
  bare {computation: order} dict (no identity; an error if `identity` requires it).
  Nothing is applied; this is the dry run an agent uses to score a proposal.
  """
  plan = plan if isinstance(plan, Plan) else Plan(computations=plan)
  violations = tuple(verify_against(plan, module, baseline, profile, identity)) + tuple(
      validate_plan(module, plan.computations, baseline=baseline, check_device_groups=check_device_groups)
  )
  before = analyze_module(module, baseline, while_bodies=while_bodies, sm_gate=sm_gate)
  if errors(list(violations)):
    return PlanReport(violations, before)
  orders = {name: plan.computations.get(name, order) for name, order in baseline.items()}
  after = analyze_module(module, orders, while_bodies=while_bodies, sm_gate=sm_gate)
  after_by_name = {c.name: c for c in after.computations}
  deltas = tuple(
      ComputationDelta(c.name, c.static_exposed_us, a.static_exposed_us, c.sim_stall_us, a.sim_stall_us)
      for c in before.computations
      if (a := after_by_name.get(c.name)) is not None and orders[c.name] != baseline[c.name]
  )
  return PlanReport(violations, before, after, deltas)
