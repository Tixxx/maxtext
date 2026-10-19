"""Static overlap / exposed-time metric.

Port of `_prefix_costs_excluding` and `_total_exposed_us`. For an async pair with
profiled latency L, overlap is the summed cost of instructions strictly between
start and done, and deficit = max(0, L - overlap). Total exposed time is the
metric that decides whether a move is a net win.

Missing costs count as 0 (free); use `coverage_gaps` to find them rather than
trusting a low exposed number.
"""

from __future__ import annotations

from dataclasses import dataclass

from maxtext.utils.agent_assisted_scheduling.schedule_model import AsyncRole, Computation, positions


@dataclass(frozen=True)
class CollectiveExposure:
  start: str
  done: str
  latency_us: float
  overlap_us: float

  @property
  def deficit_us(self) -> float:
    return max(0.0, self.latency_us - self.overlap_us)


def prefix_costs(comp: Computation, order: list[str]) -> list[float]:
  """prefix[i] = summed cost of order[:i]."""
  prefix = [0.0] * (len(order) + 1)
  for i, name in enumerate(order):
    prefix[i + 1] = prefix[i] + comp[name].cost_us
  return prefix


def collective_exposures(comp: Computation, order: list[str]) -> list[CollectiveExposure]:
  """Exposure of every async pair with a positive profiled latency, in schedule order."""
  pos = positions(order)
  prefix = prefix_costs(comp, order)
  out = []
  for start, done in comp.async_pairs():
    latency = comp[start].latency_us
    if latency is None or latency <= 0 or start not in pos or done not in pos:
      continue
    s, d = pos[start], pos[done]
    overlap = prefix[d] - prefix[s + 1] if d > s + 1 else 0.0
    out.append(CollectiveExposure(start, done, latency, overlap))
  out.sort(key=lambda e: pos[e.start])
  return out


def total_exposed_us(comp: Computation, order: list[str]) -> float:
  """Sum of positive deficits over every collective with a known latency."""
  return sum(e.deficit_us for e in collective_exposures(comp, order))


def coverage_gaps(comp: Computation, order: list[str]) -> list[str]:
  """Async starts without a profiled latency, and non-trivial ops with zero cost
  that sit inside some async window (their cost is silently treated as free)."""
  pos = positions(order)
  gaps = [
      f"{s}: no profiled latency" for s, _ in comp.async_pairs()
      if not comp[s].latency_us or comp[s].latency_us <= 0
  ]
  windows = [(pos[s], pos[d]) for s, d in comp.async_pairs() if s in pos and d in pos]
  for name in order:
    inst = comp[name]
    if inst.cost_us > 0 or inst.is_trivially_movable or inst.async_role is not AsyncRole.NONE:
      continue
    if any(s < pos[name] < d for s, d in windows):
      gaps.append(f"{name}: zero cost inside an async window")
  return gaps
