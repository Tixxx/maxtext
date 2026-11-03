"""Bounded local search for a better order of one computation.

The deterministic counterpart of the playbook's moves, for use offline or as a
starting point for an agent. It works window by window: for each collective that
is still exposed (priority collectives first, then by deficit) it tries

  hoist start  move the start earlier,
  pull         move a heavy instruction that sits after the done into the window,
  sink done    move the done later,

where every move relocates an instruction together with what has to move with it:

  hoist  the instruction and its transitive operands / control predecessors that
         sit at or after the target, placed at the target;
  sink   the instruction and its transitive users that sit at or before the
         target, placed just after it.

Both keep every operand and control edge satisfied, so XLA never has to repair the
result. A candidate is kept only if it

  * has no legality error,
  * adds no synchronous-conversion warning (a start/done pair with no real op
    between them),
  * regresses no priority collective (`priority_regressions`), and
  * lowers `static + sim_weight * stall`, where static is the summed deficit and
    stall is the one-queue comm-stream simulation (which sees queueing that the
    static window sum does not).

Optimizing the static total alone can raise the stall (a recorded body went from
13.0 ms to 15.7 ms of stall for 70 us of static gain); keep `sim_weight` > 0.

When no window can be helped directly, a general pass tries every heavy or async
instruction against sampled anchors, so the search does not stall in a local optimum
of the window moves.

Speed: candidates are scored with array arithmetic (no per-instruction Python
objects), the stall simulation only sees the instructions that have a cost or are
async, and only the few candidates with the best static score go through the full
simulation and the legality checks. A computation of 14,000 instructions takes
about a second per pass.

The search stops at the wall-clock budget or when a full pass finds nothing. Results
depend on how far the budget lets it get; compute a plan once and ship it, never per
rank.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Optional

import numpy as np

from maxtext.utils.agent_assisted_scheduling.legality import errors, validate_order
from maxtext.utils.agent_assisted_scheduling.priority import DEFAULT_PRIORITY_PREFIXES, is_priority, priority_regressions
from maxtext.utils.agent_assisted_scheduling.schedule_model import AsyncRole, Computation
from maxtext.utils.agent_assisted_scheduling.simulate import DEFAULT_SM_GATE_MIN_US

DEFAULT_SIM_WEIGHT = 0.3
DEFAULT_TIME_BUDGET_S = 120.0
HEAVY_US = 20.0  # an instruction at least this costly is a placement anchor
MOVABLE_US = 5.0  # ... and one at least this costly is worth pulling into a window
HOIST_SAMPLES = 10  # start positions tried per collective
PULL_SAMPLES = 12  # heavy instructions tried per collective
SINK_SAMPLES = 8  # done positions tried per collective
PULL_LOOKAHEAD = 300  # positions after the done searched for heavy instructions
MAX_BLOCK = 256  # a move that drags more instructions than this is skipped
FULL_SCORE_TOP = 4  # candidates per collective that get the full simulation
GENERAL_TARGETS = 40  # anchors sampled per instruction in the general pass
MIN_GAIN_US = 0.5
_GATE_PASSES = 3


@dataclass(frozen=True)
class OptimizeResult:
  order: list[str]
  static_before_us: float
  static_after_us: float
  stall_before_us: float
  stall_after_us: float
  moves: int
  seconds: float
  stopped: str  # "converged" | "time budget"


class _Fast:
  """Order-independent arrays for one computation, and the two metrics on an id order."""

  def __init__(self, comp: Computation, sm_gate: bool, sm_gate_min_us: float, prefixes: tuple[str, ...]):
    self.comp = comp
    self.names = list(comp.instructions)
    self.idx = {n: i for i, n in enumerate(self.names)}
    insts = [comp[n] for n in self.names]
    n = len(insts)
    self.cost_l = [i.cost_us for i in insts]
    self.cost = np.array(self.cost_l, dtype=np.float64)
    self.lat: list[Optional[float]] = [None] * n
    for i, inst in enumerate(insts):
      if inst.async_role is AsyncRole.START and inst.uses_comm_stream and inst.latency_us is not None:
        self.lat[i] = inst.latency_us
    self.done_start = [-1] * n
    for i, inst in enumerate(insts):
      if inst.async_role is AsyncRole.DONE and inst.operands and inst.operands[0] in self.idx:
        s = self.idx[inst.operands[0]]
        if self.lat[s] is not None:
          self.done_start[i] = s
    self.gate_l = [bool(sm_gate and inst.sm_gate and inst.cost_us >= sm_gate_min_us) for inst in insts]
    self.any_gate = any(self.gate_l)
    self.rel = np.array(
        [c > 0 or l is not None or d >= 0 for c, l, d in zip(self.cost_l, self.lat, self.done_start)], dtype=bool
    )
    self.anchor = np.array(
        [c >= HEAVY_US or inst.async_role is not AsyncRole.NONE for c, inst in zip(self.cost_l, insts)], dtype=bool
    )
    self.deps = [
        [self.idx[d] for d in dict.fromkeys(inst.operands + inst.control_predecessors) if d in self.idx] for inst in insts
    ]
    self.users: list[list[int]] = [[] for _ in range(n)]
    for i, ds in enumerate(self.deps):
      for d in ds:
        self.users[d].append(i)
    pairs = [(s, d) for s, d in comp.async_pairs() if (comp[s].latency_us or 0) > 0]
    self.ps = np.array([self.idx[s] for s, _ in pairs], dtype=np.int64)
    self.pd = np.array([self.idx[d] for _, d in pairs], dtype=np.int64)
    self.pl = np.array([comp[s].latency_us for s, _ in pairs], dtype=np.float64)
    self.priority = np.array([is_priority(comp[s], prefixes) for s, _ in pairs], dtype=bool)
    self.asyncish = np.array([inst.async_role is not AsyncRole.NONE for inst in insts], dtype=bool)

  def positions(self, order: np.ndarray) -> np.ndarray:
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    return pos

  def deficits(self, order: np.ndarray, pos: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-pair deficit (us), the static metric of cost.py."""
    pos = self.positions(order) if pos is None else pos
    prefix = np.concatenate(([0.0], np.cumsum(self.cost[order])))
    s, d = pos[self.ps], pos[self.pd]
    overlap = np.where(d > s + 1, prefix[d] - prefix[s + 1], 0.0)
    return np.maximum(0.0, self.pl - overlap)

  def stall(self, order: np.ndarray) -> float:
    """Total stall of the one-queue comm-stream simulation (simulate.py, on the relevant instructions only)."""
    events = order[self.rel[order]].tolist()
    lat, done_start, cost, gate = self.lat, self.done_start, self.cost_l, self.gate_l
    intervals: list[tuple[float, float]] = []
    total = 0.0
    for _ in range(_GATE_PASSES if self.any_gate else 1):
      t = comm = total = 0.0
      end: dict[int, float] = {}
      found: list[tuple[float, float]] = []
      for i in events:
        s = done_start[i]
        if s >= 0:
          e = end.get(s)
          if e is not None and e > t:
            total += e - t
            t = e
        elif lat[i] is not None:
          begin = max(t, comm)
          for gs, ge in intervals:
            if gs < begin < ge:
              begin = ge
          comm = begin + lat[i]
          end[i] = comm
        else:
          if gate[i]:
            found.append((t, t + cost[i]))
          t += cost[i]
      if not self.any_gate or found == intervals:
        break
      intervals = found
    return total


def _closure(start: int, rel: list[list[int]], keep) -> Optional[set[int]]:
  """`start` plus everything reachable through `rel` while `keep` holds; None if it grows past MAX_BLOCK."""
  seen = {start}
  stack = [start]
  while stack:
    for nxt in rel[stack.pop()]:
      if nxt not in seen and keep(nxt):
        seen.add(nxt)
        if len(seen) > MAX_BLOCK:
          return None
        stack.append(nxt)
  return seen


def _splice(order: np.ndarray, moving: set[int], k: int, n: int) -> np.ndarray:
  """`order` with `moving` (kept in their relative order) inserted after the first `k` other instructions."""
  mask = np.zeros(n, dtype=bool)
  mask[list(moving)] = True
  m = mask[order]
  rest, block = order[~m], order[m]
  return np.concatenate((rest[:k], block, rest[k:]))


def _hoist(fast: _Fast, order: np.ndarray, pos: np.ndarray, x: int, at: int) -> Optional[np.ndarray]:
  moving = _closure(x, fast.deps, lambda z: pos[z] >= at)
  return None if moving is None else _splice(order, moving, at, len(order))


def _sink(fast: _Fast, order: np.ndarray, pos: np.ndarray, x: int, at: int) -> Optional[np.ndarray]:
  """Move `x` and its users at or before position `at` to just after the instruction there."""
  moving = _closure(x, fast.users, lambda z: pos[z] <= at)
  if moving is None or int(order[at]) in moving:
    return None
  return _splice(order, moving, at + 1 - len(moving), len(order))


def _spread(values: np.ndarray, k: int) -> list[int]:
  """Up to `k` entries of `values`, evenly spread (first and last included)."""
  if len(values) <= k:
    return values.tolist()
  return values[np.linspace(0, len(values) - 1, k).round().astype(int)].tolist()


def _candidates(fast: _Fast, order: np.ndarray, pos: np.ndarray, pair: int, rng: random.Random):
  """Orders to try for the window of collective `pair`: hoist its start, pull heavy compute in, sink its done."""
  s, d = int(pos[fast.ps[pair]]), int(pos[fast.pd[pair]])
  anchors = np.nonzero(fast.anchor[order])[0]
  before = anchors[anchors < s]
  for at in _spread(before, HOIST_SAMPLES):
    cand = _hoist(fast, order, pos, int(order[s]), at)
    if cand is not None:
      yield cand
  tail = order[d + 1 : d + 1 + PULL_LOOKAHEAD]
  heavy = [i for i in tail.tolist() if fast.cost_l[i] >= MOVABLE_US and not fast.asyncish[i]]
  heavy.sort(key=lambda i: -fast.cost_l[i])
  pool = heavy[: 3 * PULL_SAMPLES]
  for x in rng.sample(pool, min(PULL_SAMPLES, len(pool))) if len(pool) > PULL_SAMPLES else pool:
    cand = _hoist(fast, order, pos, x, s + 1)
    if cand is not None:
      yield cand
  after = anchors[anchors > d]
  for at in _spread(after[:PULL_LOOKAHEAD], SINK_SAMPLES):
    cand = _sink(fast, order, pos, int(order[d]), at)
    if cand is not None:
      yield cand


def _general_candidates(fast: _Fast, order: np.ndarray, pos: np.ndarray, x: int, rng: random.Random):
  """Move `x` (with its closure) to sampled anchor positions on either side of where it is."""
  here = int(pos[x])
  anchors = np.nonzero(fast.anchor[order])[0]
  if len(anchors) > GENERAL_TARGETS:
    anchors = np.sort(rng.sample(anchors.tolist(), GENERAL_TARGETS))
  for p in anchors.tolist():
    if p < here:
      cand = _hoist(fast, order, pos, x, p)
    elif p > here + 1:
      cand = _sink(fast, order, pos, x, p - 1)
    else:
      continue
    if cand is not None:
      yield cand


def optimize_computation(
    comp: Computation,
    baseline: list[str],
    *,
    prefixes: tuple[str, ...] = DEFAULT_PRIORITY_PREFIXES,
    sm_gate: bool = True,
    sim_weight: float = DEFAULT_SIM_WEIGHT,
    time_budget_s: float = DEFAULT_TIME_BUDGET_S,
    seed: int = 0,
    start_order: Optional[list[str]] = None,
    sm_gate_min_us: float = DEFAULT_SM_GATE_MIN_US,
) -> OptimizeResult:
  """Improve `baseline` (XLA's order) within the budget. `start_order` resumes from a prior result."""
  t0 = time.monotonic()
  rng = random.Random(seed)
  fast = _Fast(comp, sm_gate, sm_gate_min_us, prefixes)
  names = fast.names
  n = len(names)
  to_ids = lambda order: np.array([fast.idx[x] for x in order], dtype=np.int64)
  base_ids = to_ids(baseline)
  sync0 = sum(1 for v in validate_order(comp, baseline, baseline=baseline) if v.code == "SYNC_CONVERSION")

  def admissible(ids: np.ndarray) -> bool:
    order = [names[i] for i in ids]
    found = validate_order(comp, order, baseline=baseline)
    if errors(found) or sum(1 for v in found if v.code == "SYNC_CONVERSION") > sync0:
      return False
    return not priority_regressions(comp, baseline, order, prefixes, sm_gate=sm_gate)

  state = {
      "order": to_ids(start_order) if start_order else base_ids.copy(),
  }
  state["pos"] = fast.positions(state["order"])
  state["deficits"] = fast.deficits(state["order"], state["pos"])
  state["static"] = float(state["deficits"].sum())
  state["stall"] = fast.stall(state["order"])
  moves = 0
  stopped = "converged"
  timed_out = lambda: time.monotonic() - t0 > time_budget_s

  def consider(candidates) -> bool:
    """Take the best admissible candidate that lowers the score; True if one was taken."""
    scored = []
    for cand in candidates:
      static = float(fast.deficits(cand).sum())
      if static < state["static"] - MIN_GAIN_US:
        scored.append((static, len(scored), cand))
    scored.sort(key=lambda item: item[:2])
    best: Optional[tuple[float, np.ndarray, float, float]] = None
    bar = state["static"] + sim_weight * state["stall"] - MIN_GAIN_US
    for static, _, cand in scored[:FULL_SCORE_TOP]:
      stall = fast.stall(cand)
      score = static + sim_weight * stall
      if score < bar and (best is None or score < best[0]) and admissible(cand):
        best = (score, cand, static, stall)
    if best is None:
      return False
    _, state["order"], state["static"], state["stall"] = best
    state["pos"] = fast.positions(state["order"])
    state["deficits"] = fast.deficits(state["order"], state["pos"])
    return True

  while True:
    if timed_out():
      stopped = "time budget"
      break
    improved = False
    # Window by window: priority collectives first, then the largest deficit.
    deficits = state["deficits"]
    for pair in [int(p) for p in np.lexsort((-deficits, ~fast.priority)) if deficits[p] > MIN_GAIN_US]:
      if timed_out():
        break
      if state["deficits"][pair] > MIN_GAIN_US and consider(_candidates(fast, state["order"], state["pos"], pair, rng)):
        moves += 1
        improved = True
    if improved:
      continue
    # Nothing helps a window directly: try any heavy or async instruction against sampled anchors.
    movers = [i for i in range(n) if fast.cost_l[i] >= MOVABLE_US or fast.asyncish[i]]
    movers.sort(key=lambda i: -fast.cost_l[i])
    for x in movers:
      if timed_out():
        break
      if consider(_general_candidates(fast, state["order"], state["pos"], x, rng)):
        moves += 1
        improved = True
    if not improved:
      break
  return OptimizeResult(
      order=[names[i] for i in state["order"]],
      static_before_us=float(fast.deficits(base_ids).sum()),
      static_after_us=state["static"],
      stall_before_us=fast.stall(base_ids),
      stall_after_us=state["stall"],
      moves=moves,
      seconds=time.monotonic() - t0,
      stopped=stopped,
  )
